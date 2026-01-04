# -*- coding: utf-8 -*-
"""
run_my_fvc5.py

DeepPrint (Tex+Minu) on FVC-style folder with TTA + robust view-pair scoring.

Only changes vs your best baseline:
A) Disable CLAHE (default OFF). Use --clahe-enable to turn it on.
C) Use mask centroid for crop center (option --center-mode mask).
   - If roi-mode != NONE and center-mode not specified, default to "mask".
   - If no mask available, fallback to OTSU centroid.

Keep everything else unchanged:
- rotate -> blur -> otsu -> centroid -> crop_size -> pad_and_resize
- optional mask apply: none/white/black (no crop_pad / bbox in this v5)
"""

import os
import sys
import argparse
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from sklearn.metrics import roc_curve

# Make flx importable from project root
sys.path.append(os.getcwd())

from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu
from flx.setup.datasets import get_fvc2004_db1a
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
from flx.data.image_loader import FVC2004Loader

# pyfing (you copied it into this repo)
try:
    import pyfing.simple_api as pf
except Exception:
    pf = None


# ============================================================
# JSONL helpers
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


# ============================================================
# Metrics
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
# Config
# ============================================================

@dataclass
class PreprocCfg:
    # base preproc
    clahe_enable: bool = False
    clahe_clip: float = 3.0
    clahe_grid_x: int = 8
    clahe_grid_y: int = 8
    blur_ksize: int = 5
    rot_border: int = 255
    fill: float = 1.0

    # TTA view params
    crop_size: int = 380
    angle: float = 0.0

    # ROI / mask params
    roi_mode: str = "NONE"              # NONE | PYFING_GMFS | PYFING_SUFS
    mask_apply_mode: str = "none"       # none | white | black
    dpi: int = 500

    # crop center selection
    center_mode: str = "otsu"           # otsu | mask

    # debug
    debug_save_dir: Optional[str] = None


# ============================================================
# ROI helpers
# ============================================================

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)

def get_pyfing_mask(img_u8: np.ndarray, dpi: int, roi_mode: str) -> np.ndarray:
    """
    roi_mode: PYFING_GMFS | PYFING_SUFS
    returns uint8 mask 0/255
    """
    if pf is None:
        raise RuntimeError("pyfing import failed. Please ensure ./pyfing exists and is importable.")

    roi_mode = roi_mode.upper().strip()
    if roi_mode == "PYFING_GMFS":
        m = pf.fingerprint_segmentation(img_u8, dpi=dpi, method="GMFS")
    elif roi_mode == "PYFING_SUFS":
        m = pf.fingerprint_segmentation(img_u8, dpi=dpi, method="SUFS")
    else:
        raise ValueError(f"Unsupported roi_mode for pyfing: {roi_mode}")

    m = np.asarray(m)
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8) * 255
    else:
        m = (m > 0).astype(np.uint8) * 255
    return m

def apply_mask_to_image(img: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """
    mode: none|white|black
    img uint8 gray
    mask uint8 0/255
    """
    mode = mode.lower().strip()
    if mode == "none":
        return img
    out = img.copy()
    if mode == "white":
        out[mask == 0] = 255
        return out
    if mode == "black":
        out[mask == 0] = 0
        return out
    raise ValueError(f"mask_apply_mode must be none/white/black, got: {mode}")

def mask_centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cy = int(np.round(ys.mean()))
    cx = int(np.round(xs.mean()))
    return cy, cx


# ============================================================
# Loader
# ============================================================

def make_loader(cfg: PreprocCfg):
    """
    Pipeline:
    1) read grayscale
    2) CLAHE (optional, default OFF)
    3) rotate
    4) get mask (optional via pyfing)
    5) apply mask (none/white/black)
    6) choose crop center:
       - center_mode=mask -> use mask centroid (fallback to otsu if mask empty)
       - center_mode=otsu -> blur+otsu centroid
    7) crop square crop_size
    8) pad+resize to DeepPrint input
    """
    @staticmethod
    def _load(filepath: str):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {filepath}")
        img = _ensure_uint8(img)

        # 2) CLAHE (optional)
        if cfg.clahe_enable:
            clahe = cv2.createCLAHE(
                clipLimit=float(cfg.clahe_clip),
                tileGridSize=(int(cfg.clahe_grid_x), int(cfg.clahe_grid_y)),
            )
            img_p = clahe.apply(img)
        else:
            img_p = img

        # 3) rotate
        if float(cfg.angle) != 0.0:
            h, w = img_p.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), float(cfg.angle), 1.0)
            img_r = cv2.warpAffine(img_p, M, (w, h), borderValue=int(cfg.rot_border))
        else:
            img_r = img_p

        # 4) mask (optional)
        mask = None
        roi_mode = (cfg.roi_mode or "NONE").upper().strip()
        if roi_mode != "NONE":
            mask = get_pyfing_mask(img_r, dpi=int(cfg.dpi), roi_mode=roi_mode)

        # 5) apply mask to image (optional)
        img_m = img_r
        if mask is not None:
            img_m = apply_mask_to_image(img_r, mask, mode=cfg.mask_apply_mode)

        # 6) choose crop center
        center_mode = (cfg.center_mode or "otsu").lower().strip()

        cy, cx = img_m.shape[0] // 2, img_m.shape[1] // 2

        used_mask_center = False
        if center_mode == "mask" and mask is not None:
            mc = mask_centroid(mask)
            if mc is not None:
                cy, cx = mc
                used_mask_center = True

        if not used_mask_center:
            # fallback to OTSU centroid (your original logic)
            k = int(cfg.blur_ksize)
            if k % 2 == 0:
                k += 1
            blur = cv2.GaussianBlur(img_m, (k, k), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = np.argwhere(th > 0)
            if len(coords) > 0:
                cy, cx = coords.mean(0).astype(int)

        # 7) crop square
        cs = int(cfg.crop_size)
        cs = min(cs, img_m.shape[0], img_m.shape[1])
        sy = max(0, min(cy - cs // 2, img_m.shape[0] - cs))
        sx = max(0, min(cx - cs // 2, img_m.shape[1] - cs))
        img_crop = img_m[sy:sy + cs, sx:sx + cs]

        # 8) DeepPrint input formatting
        out = pad_and_resize_to_deepprint_input_size(img_crop, fill=float(cfg.fill))
        return out

    return _load


# ============================================================
# Model loading
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
# view-pair robust scoring
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
# Feature extraction (TTA)
# ============================================================

@torch.no_grad()
def extract_views_tex_minu(
    extractor,
    data_dir: str,
    crop_sizes: List[int],
    angles: List[float],
    preproc_base: PreprocCfg,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:

    tex_views_list: List[torch.Tensor] = []
    minu_views_list: List[torch.Tensor] = []

    for cs in crop_sizes:
        for ang in angles:
            cfg = PreprocCfg(**asdict(preproc_base))
            cfg.crop_size = int(cs)
            cfg.angle = float(ang)

            print(
                f"[View] crop_size={cfg.crop_size} angle={cfg.angle} | "
                f"CLAHE={'on' if cfg.clahe_enable else 'off'} "
                f"(clip={cfg.clahe_clip} grid=({cfg.clahe_grid_x},{cfg.clahe_grid_y})) | "
                f"ROI={cfg.roi_mode} apply={cfg.mask_apply_mode} center={cfg.center_mode} dpi={cfg.dpi} | "
                f"blur={cfg.blur_ksize} border={cfg.rot_border} fill={cfg.fill}"
            )

            FVC2004Loader._load_image = make_loader(cfg)
            ds = get_fvc2004_db1a(data_dir)
            if len(ds) == 0:
                raise ValueError("Dataset is empty. Check --data path.")

            tex, minu = extractor.extract(ds)

            t = tex._array if torch.is_tensor(tex._array) else torch.from_numpy(tex._array)
            m = minu._array if torch.is_tensor(minu._array) else torch.from_numpy(minu._array)

            tex_views_list.append(F.normalize(t.to(device), p=2, dim=1))
            minu_views_list.append(F.normalize(m.to(device), p=2, dim=1))

    tex_views = torch.stack(tex_views_list, dim=1)   # [800,V,D]
    minu_views = torch.stack(minu_views_list, dim=1) # [800,V,D]
    return tex_views, minu_views


# ============================================================
# Core evaluation (8-fold 1:1)
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
        enroll_idx = (torch.arange(100, device=device) * 8 + k)  # [100]
        enroll_labels = torch.arange(100, device=device)         # [100]

        probe_idx_list = []
        probe_label_list = []
        for f in range(100):
            base = f * 8
            for imp in range(8):
                if imp == k:
                    continue
                probe_idx_list.append(base + imp)
                probe_label_list.append(f)

        probe_idx = torch.tensor(probe_idx_list, device=device, dtype=torch.long)      # [700]
        probe_labels = torch.tensor(probe_label_list, device=device, dtype=torch.long) # [700]

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
# CLI
# ============================================================

def parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_float_list(s: str) -> List[float]:
    s = (s or "").strip()
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
    ap.add_argument("--crop-sizes", default="340,370,400,430")
    ap.add_argument("--angle", dest="angles", action="append", type=float,
                    help="Use multiple times: --angle -15 --angle 0 --angle 15")

    # CLAHE (default OFF)
    ap.add_argument("--clahe-enable", action="store_true", help="Enable CLAHE (default: OFF)")
    ap.add_argument("--clahe-clip", type=float, default=3.0)
    ap.add_argument("--clahe-grid", default="8,8")

    # other preproc
    ap.add_argument("--blur-ksize", type=int, default=5)
    ap.add_argument("--rot-border", type=int, default=255)
    ap.add_argument("--fill", type=float, default=1.0)

    # ROI / mask
    ap.add_argument("--roi-mode", default="NONE",
                    choices=["NONE", "PYFING_GMFS", "PYFING_SUFS"])
    ap.add_argument("--mask-apply-mode", default="none",
                    choices=["none", "white", "black"])
    ap.add_argument("--dpi", type=int, default=500)

    # NEW: crop center mode
    ap.add_argument("--center-mode", default=None,
                    choices=["otsu", "mask"],
                    help="Crop center from OTSU centroid or mask centroid. "
                         "If not set: use mask when roi-mode!=NONE else otsu.")

    # fusion/scoring
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--tex-pair-topk", type=int, default=6)
    ap.add_argument("--tex-spike-clip", action="store_true", default=True)
    ap.add_argument("--no-tex-spike-clip", action="store_true", default=False)
    ap.add_argument("--tex-delta", type=float, default=0.08)
    ap.add_argument("--minu-gap-lam", type=float, default=0.35)

    # targets
    ap.add_argument("--target-fmr-main", type=float, default=2e-5)
    ap.add_argument("--target-fmrs", default="0.05,0.001,0.0001,0.00002")
        # DEBUG: center check
    ap.add_argument("--debug-center", action="store_true",
                    help="Save/print mask-center vs otsu-center for checking.")
    ap.add_argument("--debug-center-dir", default="./examples/out/dbg_center_check_v5",
                    help="Where to save center-check visualizations.")


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
    target_fmrs = parse_float_list(args.target_fmrs)

    tex_spike_clip_enable = True
    if args.no_tex_spike_clip:
        tex_spike_clip_enable = False
    elif args.tex_spike_clip:
        tex_spike_clip_enable = True

    # Decide default center_mode
    if args.center_mode is None:
        center_mode = "mask" if args.roi_mode != "NONE" else "otsu"
    else:
        center_mode = args.center_mode

    # load model once
    extractor = load_model(args.model, device=device, n_classes=8000, emb_dim=256)

    base_preproc = PreprocCfg(
        clahe_enable=bool(args.clahe_enable),
        clahe_clip=float(args.clahe_clip),
        clahe_grid_x=int(gx),
        clahe_grid_y=int(gy),
        blur_ksize=int(args.blur_ksize),
        rot_border=int(args.rot_border),
        fill=float(args.fill),

        roi_mode=args.roi_mode,
        mask_apply_mode=args.mask_apply_mode,
        dpi=int(args.dpi),
        center_mode=center_mode,
    )

    t0 = time.time()
    tex_views, minu_views = extract_views_tex_minu(
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
    elapsed = time.time() - t0

    print("\n================ 1:1 RESULT ================")
    for t in target_fmrs:
        fn = res[f"fnmr@{t:g}"]
        th = res[f"thr@{t:g}"]
        print(f"@{t:g}      FNMR={fn*100:6.2f}%  thr≈{th:.4f}")
    print("-------------------------------------------------------------")
    print(f"[@{args.target_fmr_main:g}] FNMR={res['fnmr_main']*100:.2f}% thr≈{res['thr_main']:.4f}")
    print(f"EER = {res['eer']*100:.2f}%   (threshold ≈ {res['eer_thr']:.4f})")
    print(f"[Time] {elapsed:.1f}s")
    print("=============================================================\n")


if __name__ == "__main__":
    main()
