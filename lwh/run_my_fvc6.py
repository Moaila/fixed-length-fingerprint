# -*- coding: utf-8 -*-
"""
run_my_fvc6.py

在 v3/v5 的 DeepPrint(Tex+Minu) FVC2000 Db1_a 评估脚本基础上，
把“BM3D 去噪 + pyfing 增强(GBFEN) + mask(已替换) + CLAHE(默认开)”接入到数据加载/预处理里。

预处理顺序（每个 view: crop_size + angle）：
1) 读灰度
2) CLAHE（默认开，可 --no-clahe 关闭）
3) rotate
4) BM3D 去噪（默认开，可 --no-bm3d 关闭）
5) pyfing 增强（默认开，可 --no-enhance 关闭）:
   segmentation(GMFS/SUFS) -> orientation(GBFOE) -> frequency(XSFFE) -> enhancement(GBFEN)
6) mask apply（none/white/black，建议 white）
7) OTSU centroid crop（保持与你 baseline 一致）
8) pad_and_resize_to_deepprint_input_size

注意：
- 为了避免 TF 在 RTX 5070 上崩溃，这里只禁用 TensorFlow 的 GPU，
  不影响 PyTorch 使用 GPU。
"""

import os
import sys
import argparse
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve

# 让 flx 可以被 import
sys.path.append(os.getcwd())

from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu
from flx.setup.datasets import get_fvc2004_db1a
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
from flx.data.image_loader import FVC2004Loader

# ============== 只禁用 TF 的 GPU（不影响 torch） ==============
def _disable_tf_gpu_only():
    try:
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
    except Exception:
        pass

_disable_tf_gpu_only()

# pyfing
try:
    import pyfing.simple_api as pf
except Exception:
    pf = None

# BM3D
try:
    from bm3d import bm3d as bm3d_func
except Exception:
    bm3d_func = None


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
    # CLAHE
    clahe_enable: bool = True
    clahe_clip: float = 3.0
    clahe_grid_x: int = 8
    clahe_grid_y: int = 8

    # rotate / crop
    blur_ksize: int = 5
    rot_border: int = 255
    fill: float = 1.0
    crop_size: int = 380
    angle: float = 0.0

    # ROI/mask
    roi_mode: str = "PYFING_GMFS"          # NONE | PYFING_GMFS | PYFING_SUFS
    mask_apply_mode: str = "white"         # none | white | black
    dpi: int = 500

    # BM3D
    bm3d_enable: bool = True
    bm3d_sigma: float = 25/255.0           # in [0,1]

    # enhancement
    enhance_enable: bool = True
    of_method: str = "GBFOE"
    freq_method: str = "XSFFE"
    enh_method: str = "GBFEN"


# ============================================================
# Helpers
# ============================================================

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)

def get_pyfing_mask(img_u8: np.ndarray, dpi: int, roi_mode: str) -> Optional[np.ndarray]:
    if roi_mode is None:
        return None
    rm = roi_mode.upper().strip()
    if rm == "NONE":
        return None
    if pf is None:
        raise RuntimeError("pyfing.simple_api import failed (pf is None)")

    if rm == "PYFING_GMFS":
        m = pf.fingerprint_segmentation(img_u8, dpi=dpi, method="GMFS")
    elif rm == "PYFING_SUFS":
        m = pf.fingerprint_segmentation(img_u8, dpi=dpi, method="SUFS")
    else:
        raise ValueError(f"Unsupported roi_mode: {roi_mode}")

    m = np.asarray(m)
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8) * 255
    else:
        m = (m > 0).astype(np.uint8) * 255
    return m

def apply_mask_to_image(img: np.ndarray, mask: Optional[np.ndarray], mode: str) -> np.ndarray:
    if mask is None:
        return img
    mode = (mode or "none").lower().strip()
    if mode == "none":
        return img
    out = img.copy()
    if mode == "white":
        out[mask == 0] = 255
        return out
    if mode == "black":
        out[mask == 0] = 0
        return out
    raise ValueError(f"mask_apply_mode must be none/white/black, got {mode}")

def bm3d_denoise_u8(img_u8: np.ndarray, sigma_01: float) -> np.ndarray:
    if bm3d_func is None:
        raise RuntimeError("bm3d package not available: `from bm3d import bm3d` failed.")
    x = img_u8.astype(np.float32) / 255.0
    den = bm3d_func(x, sigma_psd=float(sigma_01))
    den = np.clip(den, 0.0, 1.0)
    return (den * 255.0).round().astype(np.uint8)

def pyfing_enhance_u8(img_u8: np.ndarray, mask_u8: np.ndarray, dpi: int,
                      of_method: str, freq_method: str, enh_method: str) -> np.ndarray:
    """
    用 pyfing 的 GBFOE/XSFFE/GBFEN 做增强。
    返回 uint8 (H,W)。
    """
    if pf is None:
        raise RuntimeError("pyfing.simple_api import failed (pf is None)")

    orient = pf.orientation_field_estimation(
        img_u8, segmentation_mask=mask_u8, dpi=dpi, method=of_method
    )
    ridge_period = pf.frequency_estimation(
        img_u8, orient, segmentation_mask=mask_u8, dpi=dpi, method=freq_method
    )
    enhanced = pf.fingerprint_enhancement(
        img_u8, orient, ridge_period, segmentation_mask=mask_u8, dpi=dpi, method=enh_method
    )

    enhanced = np.asarray(enhanced)
    if enhanced.dtype == np.uint8:
        return enhanced

    # float / other -> uint8 normalize
    a, b = float(enhanced.min()), float(enhanced.max())
    if b - a < 1e-6:
        return np.zeros_like(img_u8, dtype=np.uint8)
    out = (enhanced - a) * (255.0 / (b - a))
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================
# Loader (核心：把 BM3D + 增强接入 flx 的 _load_image)
# ============================================================

def make_loader(cfg: PreprocCfg):
    @staticmethod
    def _load(filepath: str):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {filepath}")
        img = _ensure_uint8(img)

        # 1) CLAHE（默认开）
        if cfg.clahe_enable:
            clahe = cv2.createCLAHE(
                clipLimit=float(cfg.clahe_clip),
                tileGridSize=(int(cfg.clahe_grid_x), int(cfg.clahe_grid_y)),
            )
            img = clahe.apply(img)

        # 2) rotate
        if float(cfg.angle) != 0.0:
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), float(cfg.angle), 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=int(cfg.rot_border))

        # 3) BM3D（默认开）
        if cfg.bm3d_enable:
            img = bm3d_denoise_u8(img, sigma_01=float(cfg.bm3d_sigma))

        # 4) mask（用于增强 + 后续抹白）
        mask = get_pyfing_mask(img, dpi=int(cfg.dpi), roi_mode=cfg.roi_mode)

        # 5) pyfing 增强（默认开；需要 mask）
        if cfg.enhance_enable and mask is not None:
            img = pyfing_enhance_u8(
                img, mask, dpi=int(cfg.dpi),
                of_method=cfg.of_method,
                freq_method=cfg.freq_method,
                enh_method=cfg.enh_method
            )

        # 6) mask apply（white/black/none）
        img_m = apply_mask_to_image(img, mask, mode=cfg.mask_apply_mode)

        # 7) legacy crop by OTSU centroid（与你 baseline 对齐）
        k = int(cfg.blur_ksize)
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur(img_m, (k, k), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.argwhere(th > 0)
        cy, cx = img_m.shape[0] // 2, img_m.shape[1] // 2
        if len(coords) > 0:
            cy, cx = coords.mean(0).astype(int)

        cs = int(cfg.crop_size)
        cs = min(cs, img_m.shape[0], img_m.shape[1])
        sy = max(0, min(cy - cs // 2, img_m.shape[0] - cs))
        sx = max(0, min(cx - cs // 2, img_m.shape[1] - cs))
        img_crop = img_m[sy:sy + cs, sx:sx + cs]

        # 8) pad+resize to DeepPrint input
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
# Scoring
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
# Feature extraction
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
                f"BM3D={'on' if cfg.bm3d_enable else 'off'}(sigma={cfg.bm3d_sigma:.4f}) "
                f"ENH={'on' if cfg.enhance_enable else 'off'} "
                f"| ROI={cfg.roi_mode} apply={cfg.mask_apply_mode} dpi={cfg.dpi}"
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

    tex_views = torch.stack(tex_views_list, dim=1)
    minu_views = torch.stack(minu_views_list, dim=1)
    return tex_views, minu_views


# ============================================================
# Evaluation
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
    return out


# ============================================================
# CLI
# ============================================================

def parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_float_list(s: str) -> List[float]:
    s = (s or "").strip()
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_pair_int(s: str) -> Tuple[int, int]:
    a, b = s.split(",")
    return int(a.strip()), int(b.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    ap.add_argument("--crop-sizes", default="340,370,400,430")
    ap.add_argument("--angle", dest="angles", action="append", type=float)

    # CLAHE
    ap.add_argument("--no-clahe", action="store_true", help="Disable CLAHE (default ON)")
    ap.add_argument("--clahe-clip", type=float, default=3.0)
    ap.add_argument("--clahe-grid", default="8,8")

    # BM3D
    ap.add_argument("--no-bm3d", action="store_true", help="Disable BM3D (default ON)")
    ap.add_argument("--bm3d-sigma", type=float, default=25/255.0)

    # Enhance
    ap.add_argument("--no-enhance", action="store_true", help="Disable pyfing enhancement (default ON)")
    ap.add_argument("--of-method", default="GBFOE")
    ap.add_argument("--freq-method", default="XSFFE")
    ap.add_argument("--enh-method", default="GBFEN")

    # mask/roi
    ap.add_argument("--roi-mode", default="PYFING_GMFS", choices=["NONE", "PYFING_GMFS", "PYFING_SUFS"])
    ap.add_argument("--mask-apply-mode", default="white", choices=["none", "white", "black"])
    ap.add_argument("--dpi", type=int, default=500)

    # crop params
    ap.add_argument("--blur-ksize", type=int, default=5)
    ap.add_argument("--rot-border", type=int, default=255)
    ap.add_argument("--fill", type=float, default=1.0)

    # fusion/scoring
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--tex-pair-topk", type=int, default=6)
    ap.add_argument("--tex-delta", type=float, default=0.08)
    ap.add_argument("--no-tex-spike-clip", action="store_true", default=False)

    # targets
    ap.add_argument("--target-fmr-main", type=float, default=2e-5)
    ap.add_argument("--target-fmrs", default="0.05,0.001,0.0001,0.00002")

    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA not available, fallback to CPU.")
        device = "cpu"
    else:
        device = args.device

    crop_sizes = parse_int_list(args.crop_sizes)
    angles = args.angles if args.angles else [-15.0, 0.0, 15.0]
    gx, gy = parse_pair_int(args.clahe_grid)
    target_fmrs = parse_float_list(args.target_fmrs)

    extractor = load_model(args.model, device=device, n_classes=8000, emb_dim=256)

    base_preproc = PreprocCfg(
        clahe_enable=(not args.no_clahe),
        clahe_clip=float(args.clahe_clip),
        clahe_grid_x=int(gx),
        clahe_grid_y=int(gy),

        bm3d_enable=(not args.no_bm3d),
        bm3d_sigma=float(args.bm3d_sigma),

        enhance_enable=(not args.no_enhance),
        of_method=str(args.of_method),
        freq_method=str(args.freq_method),
        enh_method=str(args.enh_method),

        roi_mode=args.roi_mode,
        mask_apply_mode=args.mask_apply_mode,
        dpi=int(args.dpi),

        blur_ksize=int(args.blur_ksize),
        rot_border=int(args.rot_border),
        fill=float(args.fill),
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

    tex_spike_clip_enable = (not args.no_tex_spike_clip)

    res = eval_8fold_1to1(
        tex_views=tex_views,
        minu_views=minu_views,
        alpha=float(args.alpha),
        tex_pair_topk=int(args.tex_pair_topk),
        tex_spike_clip_enable=bool(tex_spike_clip_enable),
        tex_delta=float(args.tex_delta),
        target_fmrs=target_fmrs,
        target_fmr_main=float(args.target_fmr_main),
        minu_gap_lam=0.35,
    )
    elapsed = time.time() - t0

    print("\n================ 1:1 RESULT ================")
    for t in target_fmrs:
        fn = res[f"fnmr@{t:g}"]
        th = res[f"thr@{t:g}"]
        print(f"@{t:g}      FNMR={fn*100:6.2f}%  thr≈{th:.4f}")
    print("-------------------------------------------------------------")
    fnm = res["fnmr_main"] * 100
    print(f"[@{args.target_fmr_main:g}] FNMR={fnm:.2f}% thr≈{res['thr_main']:.4f}")
    print(f"EER = {res['eer']*100:.2f}%   (threshold ≈ {res['eer_thr']:.4f})")
    print(f"[Time] {elapsed:.1f}s")
    print("=============================================================\n")


if __name__ == "__main__":
    main()
