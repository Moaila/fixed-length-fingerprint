# -*- coding: utf-8 -*-
"""
DeepPrint (Tex+Minu) on FVC-style folder with TTA + robust view-pair scoring.
- Keep your current preprocessing logic unchanged
- Make preprocessing parameters adjustable at test time
- Optionally run batch configs from JSONL and output results as JSONL

Single run:
python run_my_fvc3.py --model ./models/DeepPrint_TexMinu_512/best_model.pyt --data ./data/fingerprints/FVC2000/Dbs/Db1_a --crop-sizes 340,370,400,430 --angle -15 --angle 0 --angle 15 --roi-mode PYFING_GMFS --mask-apply-mode white --debug-save-dir ./examples/out/dbg_gmfs_white

Batch run:
python run_my_fvc2.py \
  --model ./models/DeepPrint_TexMinu_512/best_model.pyt \
  --data  ./data/fingerprints/FVC2000/Dbs/Db1_a \
  --configs-jsonl ./configs.jsonl \
  --results-jsonl ./examples/out/deepprint_results.jsonl
"""

import os
import sys
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from sklearn.metrics import roc_curve

import json
import time
from datetime import datetime

# Make flx importable from project root
sys.path.append(os.getcwd())

from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu
from flx.setup.datasets import get_fvc2004_db1a
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
from flx.data.image_loader import FVC2004Loader

# ---- pyfing simple api (你贴的就是这个) ----
# 你拷贝了 pyfing/ 到当前项目，所以这里可以直接 import
try:
    from pyfing.simple_api import fingerprint_segmentation
except Exception as e:
    fingerprint_segmentation = None
    print(f"[Warn] pyfing.simple_api import failed: {e}. roi_mode=PYFING_* will not work.")


# ============================================================
#                  JSONL helpers
# ============================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def normalize_cfg(raw: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing fields from defaults, normalize types."""
    cfg = dict(defaults)
    cfg.update(raw)

    cfg["name"] = str(cfg.get("name", "unnamed"))

    cfg["crop_sizes"] = [int(x) for x in cfg["crop_sizes"]]
    cfg["angles"] = [float(x) for x in cfg["angles"]]
    cfg["clahe_grid"] = [int(cfg["clahe_grid"][0]), int(cfg["clahe_grid"][1])]
    cfg["blur_ksize"] = int(cfg["blur_ksize"])
    cfg["rot_border"] = int(cfg["rot_border"])
    cfg["fill"] = float(cfg["fill"])
    cfg["clahe_clip"] = float(cfg["clahe_clip"])

    cfg["alpha"] = float(cfg["alpha"])
    cfg["tex_pair_topk"] = int(cfg["tex_pair_topk"])
    cfg["tex_delta"] = float(cfg["tex_delta"])
    cfg["minu_gap_lam"] = float(cfg["minu_gap_lam"])
    cfg["tex_spike_clip_enable"] = bool(cfg["tex_spike_clip_enable"])

    # ROI mode
    cfg["roi_mode"] = str(cfg.get("roi_mode", "OTSU")).upper()
    # mask apply mode: "none" | "white"
    cfg["mask_apply_mode"] = str(cfg.get("mask_apply_mode", "none")).lower()

    # DPI for pyfing (GMFS/SUFS 内部会用到)
    cfg["dpi"] = int(cfg.get("dpi", 500))

    return cfg


# ============================================================
#                   Metrics helpers
# ============================================================

def compute_eer_from_scores(y_scores: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thr = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thr[idx])

def eval_at_target_from_scores(y_scores: np.ndarray, y_true: np.ndarray, target_fmr: float) -> Tuple[float, float]:
    fpr, tpr, thr = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    order = np.argsort(fpr)
    fpr, fnr, thr = fpr[order], fnr[order], thr[order]

    k = np.searchsorted(fpr, target_fmr)
    k = min(max(k, 1), len(fpr) - 1)

    w = (target_fmr - fpr[k-1]) / (fpr[k] - fpr[k-1] + 1e-12)
    fnr_i = (1 - w) * fnr[k-1] + w * fnr[k]
    thr_i = (1 - w) * thr[k-1] + w * thr[k]
    return float(fnr_i), float(thr_i)


# ============================================================
#                   Preproc Config
# ============================================================

@dataclass
class PreprocCfg:
    clahe_clip: float = 3.0
    clahe_grid_x: int = 8
    clahe_grid_y: int = 8
    blur_ksize: int = 5
    rot_border: int = 255
    fill: float = 1.0

    crop_size: int = 380
    angle: float = 0.0

    # ROI/mask
    roi_mode: str = "OTSU"          # "OTSU" | "PYFING_GMFS" | "PYFING_SUFS"
    mask_apply_mode: str = "none"   # "none" | "white"
    dpi: int = 500

    # debug / optional
    debug_save_dir: str = ""        # if non-empty, save a few preprocessed samples

    def key(self) -> str:
        return (
            f"crop{self.crop_size}_ang{self.angle}_"
            f"clahe{self.clahe_clip}_grid{self.clahe_grid_x}x{self.clahe_grid_y}_"
            f"blur{self.blur_ksize}_border{self.rot_border}_fill{self.fill}_"
            f"roi{self.roi_mode}_apply{self.mask_apply_mode}_dpi{self.dpi}"
        )


def _get_coords_from_pyfing(img_u8: np.ndarray, roi_mode: str, dpi: int) -> Optional[np.ndarray]:
    """
    roi_mode: PYFING_GMFS / PYFING_SUFS
    returns coords (N,2) where mask>0, or None if not available/fails.
    """
    if fingerprint_segmentation is None:
        return None
    try:
        if roi_mode == "PYFING_GMFS":
            mask = fingerprint_segmentation(img_u8, dpi=dpi, method="GMFS")
        elif roi_mode == "PYFING_SUFS":
            mask = fingerprint_segmentation(img_u8, dpi=dpi, method="SUFS")
        else:
            return None
        if mask is None:
            return None
        coords = np.argwhere(mask > 0)
        return coords
    except Exception as e:
        print(f"[Warn] pyfing segmentation failed ({roi_mode}): {e}")
        return None


def make_loader(cfg: PreprocCfg):
    """
    Returns a staticmethod _load(filepath) compatible with FVC2004Loader._load_image override.
    Keeps your preprocessing steps, but allows ROI coords from pyfing mask (GMFS/SUFS).
    """
    @staticmethod
    def _load(filepath: str):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {filepath}")
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # 1) CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=float(cfg.clahe_clip),
            tileGridSize=(int(cfg.clahe_grid_x), int(cfg.clahe_grid_y)),
        )
        img = clahe.apply(img)

        # 2) Rotate (TTA)
        if float(cfg.angle) != 0.0:
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), float(cfg.angle), 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=int(cfg.rot_border))

        # 3) coords from ROI
        k = int(cfg.blur_ksize)
        if k % 2 == 0:
            k += 1

        coords = None
        roi_mode = str(cfg.roi_mode).upper()

        # 3.1 pyfing segmentation
        if roi_mode in ("PYFING_GMFS", "PYFING_SUFS"):
            coords = _get_coords_from_pyfing(img, roi_mode, int(cfg.dpi))

            # 可选：把 mask 外抹白（建议先不用，做单变量）
            if coords is not None and len(coords) > 0 and str(cfg.mask_apply_mode).lower() == "white":
                # 重新算一次 mask（省事；如果你想更快，可以改 _get_coords_from_pyfing 返回 mask）
                try:
                    method = "GMFS" if roi_mode == "PYFING_GMFS" else "SUFS"
                    mask = fingerprint_segmentation(img, dpi=int(cfg.dpi), method=method)
                    if mask is not None:
                        img = img.copy()
                        img[mask == 0] = 255
                except Exception:
                    pass

        # 3.2 fallback to your original OTSU coords
        if coords is None or len(coords) == 0:
            blur = cv2.GaussianBlur(img, (k, k), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = np.argwhere(th > 0)

        # 4) centroid crop
        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        if len(coords) > 0:
            cy, cx = coords.mean(0).astype(int)

        cs_req = int(cfg.crop_size)
        cs = min(cs_req, img.shape[0], img.shape[1])

        sy = max(0, min(cy - cs // 2, img.shape[0] - cs))
        sx = max(0, min(cx - cs // 2, img.shape[1] - cs))

        cropped = img[sy:sy + cs, sx:sx + cs]

        out = pad_and_resize_to_deepprint_input_size(cropped, fill=float(cfg.fill))

        # optional debug save
                # optional debug save
        if cfg.debug_save_dir:
            os.makedirs(cfg.debug_save_dir, exist_ok=True)
            cnt = getattr(_load, "_dbg_cnt", 0)
            if cnt < 2:
                base = os.path.basename(filepath).replace("/", "_")
                save_path = os.path.join(
                    cfg.debug_save_dir,
                    f"dbg_{cnt}_{base}_roi{roi_mode}_ang{cfg.angle}_cs{cs_req}_used{cs}.png"
                )

                to_save = out
                # --- convert to numpy uint8 for cv2.imwrite ---
                if torch.is_tensor(to_save):
                    to_save = to_save.detach().cpu().numpy()

                # flx 里可能是 float[0..1] 或 [1,H,W] / [H,W,1]
                to_save = np.asarray(to_save)

                if to_save.ndim == 3:
                    # 常见情况: (1,H,W) 或 (H,W,1)
                    if to_save.shape[0] == 1:
                        to_save = to_save[0]
                    elif to_save.shape[-1] == 1:
                        to_save = to_save[..., 0]

                # 转 uint8
                if to_save.dtype != np.uint8:
                    vmin, vmax = float(to_save.min()), float(to_save.max())
                    if vmax <= 1.5:  # 大概率是 0..1
                        to_save = (to_save * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        to_save = to_save.clip(0, 255).astype(np.uint8)

                cv2.imwrite(save_path, to_save)
                _load._dbg_cnt = cnt + 1


        return out

    return _load


# ============================================================
#                   Model loading
# ============================================================

@torch.no_grad()
def load_model(model_path: str, device: str, n_classes: int = 8000, emb_dim: int = 256):
    extractor = get_DeepPrint_TexMinu(n_classes, emb_dim)
    ckpt = torch.load(model_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    md = extractor.model.state_dict()
    md.update({k: v for k, v in sd.items() if k in md and hasattr(v, "shape") and v.shape == md[k].shape})
    extractor.model.load_state_dict(md)

    extractor.model.eval()
    extractor.model.to(device)
    return extractor


# ============================================================
#          view-pair robust score functions
# ============================================================

@torch.no_grad()
def viewpair_topkmean(enroll_views: torch.Tensor, probe_views: torch.Tensor, topk: int = 6) -> torch.Tensor:
    sim = torch.einsum("nvd,mwd->nmvw", enroll_views, probe_views)  # [N,M,V,V]
    flat = sim.reshape(sim.shape[0], sim.shape[1], -1)             # [N,M,V*V]
    kk = min(int(topk), flat.shape[-1])
    return flat.topk(kk, dim=-1).values.mean(dim=-1)               # [N,M]

@torch.no_grad()
def viewpair_gapaware(enroll_views: torch.Tensor, probe_views: torch.Tensor, lam: float = 0.35) -> torch.Tensor:
    sim = torch.einsum("nvd,mwd->nmvw", enroll_views, probe_views)  # [N,M,V,V]
    flat = sim.reshape(sim.shape[0], sim.shape[1], -1)             # [N,M,V*V]
    top2 = flat.topk(2, dim=-1).values                              # [N,M,2]
    top1 = top2[..., 0]
    top2v = top2[..., 1]
    gap = top1 - top2v
    return top1 - float(lam) * gap                                   # [N,M]

@torch.no_grad()
def viewpair_top2_clip(enroll_views: torch.Tensor, probe_views: torch.Tensor, delta: float = 0.08) -> torch.Tensor:
    sim = torch.einsum("nvd,mwd->nmvw", enroll_views, probe_views)
    flat = sim.reshape(sim.shape[0], sim.shape[1], -1)
    top2 = flat.topk(2, dim=-1).values
    m1 = top2[..., 0]
    m2 = top2[..., 1]
    return torch.minimum(m1, m2 + float(delta))                    # [N,M]


# ============================================================
#                 Feature extraction (TTA)
# ============================================================

@torch.no_grad()
def extract_views_tex_minu(
    extractor,
    data_dir: str,
    crop_sizes: List[int],
    angles: List[float],
    preproc_base: PreprocCfg,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[PreprocCfg]]:
    tex_views_list, minu_views_list = [], []
    cfgs: List[PreprocCfg] = []

    for cs in crop_sizes:
        for ang in angles:
            cfg = PreprocCfg(**asdict(preproc_base))
            cfg.crop_size = int(cs)
            cfg.angle = float(ang)
            cfgs.append(cfg)

            print(f"[View] crop_size={cfg.crop_size} angle={cfg.angle} | "
                  f"ROI={cfg.roi_mode} apply={cfg.mask_apply_mode} dpi={cfg.dpi} | "
                  f"CLAHE clip={cfg.clahe_clip} grid=({cfg.clahe_grid_x},{cfg.clahe_grid_y}) "
                  f"blur={cfg.blur_ksize} border={cfg.rot_border} fill={cfg.fill}")

            FVC2004Loader._load_image = make_loader(cfg)

            ds = get_fvc2004_db1a(data_dir)
            if len(ds) == 0:
                raise ValueError("Dataset is empty. Check --data path.")

            tex, minu = extractor.extract(ds)

            t = tex._array if torch.is_tensor(tex._array) else torch.from_numpy(tex._array)
            m = minu._array if torch.is_tensor(minu._array) else torch.from_numpy(minu._array)

            tex_views_list.append(F.normalize(t.to(device), p=2, dim=1))
            minu_views_list.append(F.normalize(m.to(device), p=2, dim=1))

    tex_views = torch.stack(tex_views_list, dim=1)
    minu_views = torch.stack(minu_views_list, dim=1)
    return tex_views, minu_views, cfgs


# ============================================================
#                  Core evaluation (8-fold 1:1)
# ============================================================

@torch.no_grad()
def eval_8fold_1to1(
    tex_views: torch.Tensor,
    minu_views: torch.Tensor,
    alpha: float,
    tex_pair_topk: int,
    tex_spike_clip_enable: bool,
    tex_delta: float,
    target_fmrs: List[float],
    target_fmr_main: float,
    minu_gap_lam: float = 0.35,
) -> Dict[str, Any]:
    device = tex_views.device
    y_scores_gen = []
    y_scores_imp = []

    for k in range(8):
        enroll_idx = (torch.arange(100, device=device) * 8 + k)
        enroll_labels = torch.arange(100, device=device)

        probe_idx_list = []
        probe_label_list = []
        for f in range(100):
            base = f * 8
            for imp in range(8):
                if imp == k:
                    continue
                probe_idx_list.append(base + imp)
                probe_label_list.append(f)

        probe_idx = torch.tensor(probe_idx_list, device=device, dtype=torch.long)
        probe_labels = torch.tensor(probe_label_list, device=device, dtype=torch.long)

        enroll_tex = tex_views[enroll_idx]
        probe_tex  = tex_views[probe_idx]
        enroll_minu = minu_views[enroll_idx]
        probe_minu  = minu_views[probe_idx]

        if tex_spike_clip_enable:
            s_tex_mat = viewpair_top2_clip(enroll_tex, probe_tex, delta=tex_delta)
        else:
            s_tex_mat = viewpair_topkmean(enroll_tex, probe_tex, topk=tex_pair_topk)

        s_minu_mat = viewpair_gapaware(enroll_minu, probe_minu, lam=minu_gap_lam)

        scores_mat = s_tex_mat + float(alpha) * s_minu_mat

        mask_genuine = (enroll_labels.unsqueeze(1) == probe_labels.unsqueeze(0))
        gen_scores = scores_mat[mask_genuine].detach().cpu().numpy()
        imp_scores = scores_mat[~mask_genuine].detach().cpu().numpy()

        y_scores_gen.append(gen_scores)
        y_scores_imp.append(imp_scores)

        print(f"[Fold {k}] Gen: {len(gen_scores)} pairs | Imp: {len(imp_scores)} pairs")

    y_scores_gen = np.concatenate(y_scores_gen)
    y_scores_imp = np.concatenate(y_scores_imp)
    y_scores = np.concatenate([y_scores_gen, y_scores_imp])
    y_true = np.concatenate([np.ones_like(y_scores_gen), np.zeros_like(y_scores_imp)])

    out: Dict[str, Any] = {}
    for t in target_fmrs:
        fn, th = eval_at_target_from_scores(y_scores, y_true, t)
        out[f"fnmr@{t:g}"] = fn
        out[f"thr@{t:g}"] = th

    eer, eer_thr = compute_eer_from_scores(y_scores, y_true)
    out["eer"] = eer
    out["eer_thr"] = eer_thr

    fn_main, th_main = eval_at_target_from_scores(y_scores, y_true, target_fmr_main)
    out["fnmr_main"] = fn_main
    out["thr_main"] = th_main
    out["n_gen"] = int(len(y_scores_gen))
    out["n_imp"] = int(len(y_scores_imp))
    return out


# ============================================================
#                 CLI / main
# ============================================================

def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_float_list(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_pair_int(s: str) -> Tuple[int, int]:
    a, b = s.split(",")
    return int(a.strip()), int(b.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to best_model.pyt")
    ap.add_argument("--data", required=True, help="Dataset folder (Db1_a)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # TTA / views
    ap.add_argument("--crop-sizes", default="340,370,400,430", help="Comma-separated crop sizes for views")
    ap.add_argument("--angle", dest="angles", action="append", type=float,
                    help="Angle in degrees. Can be used multiple times, e.g. --angle -15 --angle 0 --angle 15")

    # Preproc params
    ap.add_argument("--clahe-clip", type=float, default=3.0)
    ap.add_argument("--clahe-grid", default="8,8", help="e.g. 8,8")
    ap.add_argument("--blur-ksize", type=int, default=5)
    ap.add_argument("--rot-border", type=int, default=255)
    ap.add_argument("--fill", type=float, default=1.0)

    # NEW: ROI mode
    ap.add_argument("--roi-mode", default="OTSU", choices=["OTSU", "PYFING_GMFS", "PYFING_SUFS"],
                    help="How to get ROI coords for centroid/crop: OTSU or pyfing segmentation.")
    ap.add_argument("--mask-apply-mode", default="none", choices=["none", "white"],
                    help="If roi-mode is PYFING_*, optionally paint outside-mask to white before crop.")
    ap.add_argument("--dpi", type=int, default=500, help="Fingerprint DPI for pyfing segmentation algorithms.")
    ap.add_argument("--debug-save-dir", default="", help="If set, save a few preprocessed debug images.")

    # Fusion / scoring params
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--tex-pair-topk", type=int, default=6)
    ap.add_argument("--tex-spike-clip", action="store_true", default=True)
    ap.add_argument("--no-tex-spike-clip", action="store_true", default=False)
    ap.add_argument("--tex-delta", type=float, default=0.08)
    ap.add_argument("--minu-gap-lam", type=float, default=0.35)

    # Targets
    ap.add_argument("--target-fmr-main", type=float, default=2e-5)
    ap.add_argument("--target-fmrs", default="0.05,0.001,0.0001,0.00002")

    # Batch
    ap.add_argument("--configs-jsonl", default=None, help="JSONL configs file (one JSON per line). If set, run batch.")
    ap.add_argument("--results-jsonl", default="./examples/out/deepprint_results.jsonl", help="Where to append results JSONL.")

    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA not available, fallback to CPU.")
        device = "cpu"
    else:
        device = args.device

    crop_sizes = parse_int_list(args.crop_sizes)
    angles = args.angles if args.angles else [-15.0, 0.0, 15.0]
    if not crop_sizes:
        raise ValueError("--crop-sizes is empty")
    if not angles:
        raise ValueError("--angle is empty (use --angle -15 --angle 0 --angle 15)")

    gx, gy = parse_pair_int(args.clahe_grid)

    tex_spike_clip_enable = True
    if args.no_tex_spike_clip:
        tex_spike_clip_enable = False
    elif args.tex_spike_clip:
        tex_spike_clip_enable = True

    target_fmrs = parse_float_list(args.target_fmrs)

    extractor = load_model(args.model, device=device, n_classes=8000, emb_dim=256)

    # default config template (for batch)
    default_cfg = {
        "name": "cli_default",
        "crop_sizes": crop_sizes,
        "angles": angles,
        "clahe_clip": float(args.clahe_clip),
        "clahe_grid": [int(gx), int(gy)],
        "blur_ksize": int(args.blur_ksize),
        "rot_border": int(args.rot_border),
        "fill": float(args.fill),
        "alpha": float(args.alpha),
        "tex_pair_topk": int(args.tex_pair_topk),
        "tex_spike_clip_enable": bool(tex_spike_clip_enable),
        "tex_delta": float(args.tex_delta),
        "minu_gap_lam": float(args.minu_gap_lam),
        "roi_mode": str(args.roi_mode).upper(),
        "mask_apply_mode": str(args.mask_apply_mode).lower(),
        "dpi": int(args.dpi),
    }

    # -------- batch mode --------
    if args.configs_jsonl is not None:
        cfg_list_raw = read_jsonl(args.configs_jsonl)
        if not cfg_list_raw:
            raise ValueError(f"Empty configs file: {args.configs_jsonl}")

        print(f"[Batch] Loaded {len(cfg_list_raw)} configs from {args.configs_jsonl}")
        for i, raw in enumerate(cfg_list_raw):
            cfg = normalize_cfg(raw, default_cfg)

            print("\n=============================================================")
            print(f"[Batch {i+1}/{len(cfg_list_raw)}] name={cfg['name']}")
            print("=============================================================")

            pp = PreprocCfg(
                clahe_clip=cfg["clahe_clip"],
                clahe_grid_x=cfg["clahe_grid"][0],
                clahe_grid_y=cfg["clahe_grid"][1],
                blur_ksize=cfg["blur_ksize"],
                rot_border=cfg["rot_border"],
                fill=cfg["fill"],
                roi_mode=cfg["roi_mode"],
                mask_apply_mode=cfg["mask_apply_mode"],
                dpi=cfg["dpi"],
                debug_save_dir=args.debug_save_dir,
            )

            t0 = time.time()

            tex_views, minu_views, _ = extract_views_tex_minu(
                extractor=extractor,
                data_dir=args.data,
                crop_sizes=cfg["crop_sizes"],
                angles=cfg["angles"],
                preproc_base=pp,
                device=device,
            )

            res = eval_8fold_1to1(
                tex_views=tex_views,
                minu_views=minu_views,
                alpha=cfg["alpha"],
                tex_pair_topk=cfg["tex_pair_topk"],
                tex_spike_clip_enable=cfg["tex_spike_clip_enable"],
                tex_delta=cfg["tex_delta"],
                target_fmrs=target_fmrs,
                target_fmr_main=float(args.target_fmr_main),
                minu_gap_lam=cfg["minu_gap_lam"],
            )

            elapsed = time.time() - t0

            out_obj = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "model": args.model,
                "data": args.data,
                "device": device,
                "config": cfg,
                "metrics": {
                    "eer": res["eer"],
                    "eer_thr": res["eer_thr"],
                    "fnmr_main": res["fnmr_main"],
                    "thr_main": res["thr_main"],
                    "n_gen": res["n_gen"],
                    "n_imp": res["n_imp"],
                },
                "targets": {f"{t:g}": {"fnmr": res[f"fnmr@{t:g}"], "thr": res[f"thr@{t:g}"]} for t in target_fmrs},
                "elapsed_sec": elapsed,
            }

            append_jsonl(args.results_jsonl, out_obj)

            print(f"[Saved] {cfg['name']} -> {args.results_jsonl}")
            print(f"[Time] {elapsed:.1f}s | EER={res['eer']*100:.2f}% | FNMR@{args.target_fmr_main:g}={res['fnmr_main']*100:.2f}%")

        print(f"\n[Batch Done] results appended to: {args.results_jsonl}")
        return

    # -------- single run mode --------
    base_preproc = PreprocCfg(
        clahe_clip=float(args.clahe_clip),
        clahe_grid_x=int(gx),
        clahe_grid_y=int(gy),
        blur_ksize=int(args.blur_ksize),
        rot_border=int(args.rot_border),
        fill=float(args.fill),
        roi_mode=str(args.roi_mode).upper(),
        mask_apply_mode=str(args.mask_apply_mode).lower(),
        dpi=int(args.dpi),
        debug_save_dir=args.debug_save_dir,
    )

    tex_views, minu_views, _ = extract_views_tex_minu(
        extractor=extractor,
        data_dir=args.data,
        crop_sizes=crop_sizes,
        angles=angles,
        preproc_base=base_preproc,
        device=device,
    )

    print(f"[Info] Extracted views: V={tex_views.shape[1]} (crop_sizes={len(crop_sizes)} x angles={len(angles)})")

    res = eval_8fold_1to1(
        tex_views=tex_views,
        minu_views=minu_views,
        alpha=float(args.alpha),
        tex_pair_topk=int(args.tex_pair_topk),
        tex_spike_clip_enable=bool(tex_spike_clip_enable),
        tex_delta=float(args.tex_delta),
        target_fmrs=target_fmrs,
        target_fmr_main=float(args.target_fmr_main),
        minu_gap_lam=float(args.minu_gap_lam),
    )

    print("\n================ 1:1 RESULT ================")
    for t in target_fmrs:
        fn = res[f"fnmr@{t:g}"]
        th = res[f"thr@{t:g}"]
        print(f"@{t:<10g} FNMR={fn*100:6.2f}%  thr≈{th:.4f}")

    print("-------------------------------------------------------------")
    print(f"[@{args.target_fmr_main:g}] FNMR={res['fnmr_main']*100:.2f}% thr≈{res['thr_main']:.4f}")
    print(f"EER = {res['eer']*100:.2f}%   (threshold ≈ {res['eer_thr']:.4f})")
    print("=============================================================\n")


if __name__ == "__main__":
    main()