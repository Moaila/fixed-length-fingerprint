# -*- coding: utf-8 -*-
"""
DeepPrint (Tex+Minu) on FVC-style folder with TTA + robust view-pair scoring.
- Keep your current preprocessing logic unchanged
- Make preprocessing parameters adjustable at test time
- Optionally sweep configs and output CSV

Example (single run):
python run_my_fvc2.py   \
    --model ./models/DeepPrint_TexMinu_512/best_model.pyt \
    --data  ./data/fingerprints/FVC2000/Dbs/Db1_a   \
    --crop-sizes 340,370,400,430   \
    --angle -15   --angle 0   --angle 15   \
    --clahe-clip 3.0   \
    --clahe-grid 8,8   \
    --blur-ksize 5   \
    --rot-border 255   \
    --fill 1.0

Example (sweep a few presets):
  python eval_deepprint_tta_sweep.py \
    --model ./models/DeepPrint_TexMinu_512/best_model.pyt \
    --data ./data/fingerprints/FVC2000/Dbs/Db1_a \
    --sweep \
    --out-csv ./examples/out/deepprint_sweep.csv
"""

import os
import sys
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

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

import json
import time
from datetime import datetime


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
    """Fill missing fields from defaults."""
    cfg = dict(defaults)
    cfg.update(raw)

    # normalize types
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

    cfg["name"] = str(cfg.get("name", "unnamed"))
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
    fill: float = 1.0  # passed to pad_and_resize_to_deepprint_input_size (fill=1.0 -> white background)

    # NOTE: we DO NOT change your current logic
    # - still: rotate -> blur -> otsu -> centroid -> crop_size -> pad_and_resize
    # only make these values adjustable
    crop_size: int = 380
    angle: float = 0.0

    def key(self) -> str:
        return (
            f"crop{self.crop_size}_ang{self.angle}_"
            f"clahe{self.clahe_clip}_grid{self.clahe_grid_x}x{self.clahe_grid_y}_"
            f"blur{self.blur_ksize}_border{self.rot_border}_fill{self.fill}"
        )


def make_loader(cfg: PreprocCfg):
    """
    Returns a staticmethod _load(filepath) compatible with FVC2004Loader._load_image override.
    Keeps EXACTLY your preprocessing steps, just parameterized.
    """
    @staticmethod
    def _load(filepath: str):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {filepath}")

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

        # 3) Blur + OTSU threshold
        k = int(cfg.blur_ksize)
        if k % 2 == 0:
            k += 1  # must be odd
        blur = cv2.GaussianBlur(img, (k, k), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.argwhere(th > 0)

        # 4) centroid crop
        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        if len(coords) > 0:
            cy, cx = coords.mean(0).astype(int)

        cs = int(cfg.crop_size)
        cs = min(cs, img.shape[0], img.shape[1])
        sy = max(0, min(cy - cs // 2, img.shape[0] - cs))
        sx = max(0, min(cx - cs // 2, img.shape[1] - cs))
        img = img[sy:sy + cs, sx:sx + cs]

        # 5) DeepPrint input formatting
        return pad_and_resize_to_deepprint_input_size(img, fill=float(cfg.fill))

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
    """
    Returns:
      tex_views:  [800, V, D]
      minu_views: [800, V, D]
      cfgs: list of PreprocCfg for each view (length V)
    """
    tex_views_list, minu_views_list = [], []
    cfgs: List[PreprocCfg] = []

    for cs in crop_sizes:
        for ang in angles:
            cfg = PreprocCfg(**asdict(preproc_base))
            cfg.crop_size = int(cs)
            cfg.angle = float(ang)
            cfgs.append(cfg)

            print(f"[View] crop_size={cfg.crop_size} angle={cfg.angle} | "
                  f"CLAHE clip={cfg.clahe_clip} grid=({cfg.clahe_grid_x},{cfg.clahe_grid_y}) "
                  f"blur={cfg.blur_ksize} border={cfg.rot_border} fill={cfg.fill}")

            # Override loader
            FVC2004Loader._load_image = make_loader(cfg)

            ds = get_fvc2004_db1a(data_dir)
            if len(ds) == 0:
                raise ValueError("Dataset is empty. Check --data path.")

            tex, minu = extractor.extract(ds)

            t = tex._array if torch.is_tensor(tex._array) else torch.from_numpy(tex._array)
            m = minu._array if torch.is_tensor(minu._array) else torch.from_numpy(minu._array)

            tex_views_list.append(F.normalize(t.to(device), p=2, dim=1))   # [800,D]
            minu_views_list.append(F.normalize(m.to(device), p=2, dim=1))  # [800,D]

    tex_views = torch.stack(tex_views_list, dim=1)   # [800,V,D]
    minu_views = torch.stack(minu_views_list, dim=1) # [800,V,D]
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

        enroll_tex = tex_views[enroll_idx]      # [100,V,D]
        probe_tex  = tex_views[probe_idx]       # [700,V,D]
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

    y_scores_gen = np.concatenate(y_scores_gen)  # 5600
    y_scores_imp = np.concatenate(y_scores_imp)  # 554400
    y_scores = np.concatenate([y_scores_gen, y_scores_imp])
    y_true = np.concatenate([np.ones_like(y_scores_gen), np.zeros_like(y_scores_imp)])

    out: Dict[str, Any] = {}
    # FNMR at target FMRs
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
#                 Sweep presets
# ============================================================

def build_sweep_presets(base: PreprocCfg) -> List[PreprocCfg]:
    """
    Minimal, safe sweep:
    - different CLAHE clip
    - different blur ksize
    - different fill
    Note: crop_size & angle are still controlled by --crop-sizes and --angles
    """
    presets: List[PreprocCfg] = []

    # Baseline
    presets.append(PreprocCfg(**asdict(base)))

    # CLAHE variations
    for clip in [2.0, 3.0, 4.0]:
        cfg = PreprocCfg(**asdict(base))
        cfg.clahe_clip = clip
        presets.append(cfg)

    # Blur variations
    for k in [3, 5, 7]:
        cfg = PreprocCfg(**asdict(base))
        cfg.blur_ksize = k
        presets.append(cfg)

    # Fill variations (white vs black padding)
    for fill in [1.0, 0.0]:
        cfg = PreprocCfg(**asdict(base))
        cfg.fill = fill
        presets.append(cfg)

    # De-dup by key excluding crop/angle (since crop/angle are view list)
    uniq = {}
    for p in presets:
        uniq[(p.clahe_clip, p.clahe_grid_x, p.clahe_grid_y, p.blur_ksize, p.rot_border, p.fill)] = p
    return list(uniq.values())


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

def write_csv(path: str, rows: List[Dict[str, Any]]):
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Union keys
    keys = []
    keyset = set()
    for r in rows:
        for k in r.keys():
            if k not in keyset:
                keyset.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to best_model.pyt")
    ap.add_argument("--data", required=True, help="Dataset folder (Db1_a)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # TTA / views
    ap.add_argument("--crop-sizes", default="340,370,400,430", help="Comma-separated crop sizes for views")
    # ap.add_argument("--angles", default="-15,0,15", help="Comma-separated angles for views")
    ap.add_argument("--angle", dest="angles", action="append", type=float,
                help="Angle in degrees. Can be used multiple times, e.g. --angle -15 --angle 0 --angle 15")


    # Preproc params (keep your pipeline, just adjustable)
    ap.add_argument("--clahe-clip", type=float, default=3.0)
    ap.add_argument("--clahe-grid", default="8,8", help="e.g. 8,8")
    ap.add_argument("--blur-ksize", type=int, default=5)
    ap.add_argument("--rot-border", type=int, default=255)
    ap.add_argument("--fill", type=float, default=1.0)

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

    # Sweep
    ap.add_argument("--sweep", action="store_true", help="Run a few safe preproc presets and output CSV")
    ap.add_argument("--out-csv", default="./examples/out/deepprint_sweep.csv")
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

    # scoring switch
    tex_spike_clip_enable = True
    if args.no_tex_spike_clip:
        tex_spike_clip_enable = False
    elif args.tex_spike_clip:
        tex_spike_clip_enable = True

    target_fmrs = parse_float_list(args.target_fmrs)

    # Load model once
    extractor = load_model(args.model, device=device, n_classes=8000, emb_dim=256)

    # ===== default config template (used for batch) =====
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
    }

    # ===== batch mode =====
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

            # build preproc from cfg
            pp = PreprocCfg(
                clahe_clip=cfg["clahe_clip"],
                clahe_grid_x=cfg["clahe_grid"][0],
                clahe_grid_y=cfg["clahe_grid"][1],
                blur_ksize=cfg["blur_ksize"],
                rot_border=cfg["rot_border"],
                fill=cfg["fill"],
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

            # record result JSONL
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


    base_preproc = PreprocCfg(
        clahe_clip=float(args.clahe_clip),
        clahe_grid_x=int(gx),
        clahe_grid_y=int(gy),
        blur_ksize=int(args.blur_ksize),
        rot_border=int(args.rot_border),
        fill=float(args.fill),
    )

    rows: List[Dict[str, Any]] = []

    preproc_presets = [base_preproc]
    if args.sweep:
        preproc_presets = build_sweep_presets(base_preproc)

    for pi, pp in enumerate(preproc_presets):
        print("\n=============================================================")
        print(f"[Preset {pi+1}/{len(preproc_presets)}] {pp}")
        print("=============================================================\n")

        # Extract embeddings for all views under this preset
        tex_views, minu_views, cfgs = extract_views_tex_minu(
            extractor=extractor,
            data_dir=args.data,
            crop_sizes=crop_sizes,
            angles=angles,
            preproc_base=pp,
            device=device,
        )

        V = tex_views.shape[1]
        print(f"[Info] Extracted views: V={V} (crop_sizes={len(crop_sizes)} x angles={len(angles)})")

        # Evaluate
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

        # Print summary (same style as you like)
        print("\n================ 1:1 RESULT ================")
        for t in target_fmrs:
            fn = res[f"fnmr@{t:g}"]
            th = res[f"thr@{t:g}"]
            name = f"@{t:g}"
            print(f"{name:<10} FNMR={fn*100:6.2f}%  thr≈{th:.4f}")

        print("-------------------------------------------------------------")
        print(f"[@{args.target_fmr_main:g}] FNMR={res['fnmr_main']*100:.2f}% thr≈{res['thr_main']:.4f}")
        print(f"EER = {res['eer']*100:.2f}%   (threshold ≈ {res['eer_thr']:.4f})")
        print("=============================================================\n")

        # Record row for CSV
        row = {}
        row.update({
            "preset_index": pi,
            "device": device,
            "crop_sizes": args.crop_sizes,
            "angles": ",".join([str(a) for a in angles]),
            "alpha": args.alpha,
            "tex_pair_topk": args.tex_pair_topk,
            "tex_spike_clip_enable": tex_spike_clip_enable,
            "tex_delta": args.tex_delta,
            "minu_gap_lam": args.minu_gap_lam,
            "target_fmr_main": args.target_fmr_main,
        })
        row.update({
            "clahe_clip": pp.clahe_clip,
            "clahe_grid_x": pp.clahe_grid_x,
            "clahe_grid_y": pp.clahe_grid_y,
            "blur_ksize": pp.blur_ksize,
            "rot_border": pp.rot_border,
            "fill": pp.fill,
        })
        # metrics
        for t in target_fmrs:
            row[f"fnmr@{t:g}"] = res[f"fnmr@{t:g}"]
            row[f"thr@{t:g}"] = res[f"thr@{t:g}"]
        row["fnmr_main"] = res["fnmr_main"]
        row["thr_main"] = res["thr_main"]
        row["eer"] = res["eer"]
        row["eer_thr"] = res["eer_thr"]
        row["n_gen"] = res["n_gen"]
        row["n_imp"] = res["n_imp"]
        rows.append(row)

    if args.sweep:
        write_csv(args.out_csv, rows)
        print(f"[Saved] sweep results -> {args.out_csv}")

if __name__ == "__main__":
    main()
