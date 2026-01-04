# -*- coding: utf-8 -*-
"""
run_my_fvc4.py

DeepPrint (Tex+Minu) on FVC-style folder with TTA + robust view-pair scoring.
Adds ROI/mask modes using pyfing segmentation + different mask application strategies:
- none / white / black
- crop_pad: crop by mask bbox (with margin) then pad/resize
- bbox_crop: same as crop_pad (alias; kept for clarity)
- mask morphology: dilate/erode/open/close to control ROI boundary

Also supports:
- batch mode with configs.jsonl (one JSON per line)
- append results to results.jsonl
- debug save intermediates

Example (single run):
python run_my_fvc4.py \
  --model ./models/DeepPrint_TexMinu_512/best_model.pyt \
  --data  ./data/fingerprints/FVC2000/Dbs/Db1_a \
  --crop-sizes 340,370,400,430 \
  --angle -15 --angle 0 --angle 15 \
  --roi-mode PYFING_GMFS \
  --mask-apply-mode crop_pad \
  --bbox-margin 20 \
  --mask-morph dilate --mask-morph-ksize 5 --mask-morph-iter 1 \
  --debug-save-dir ./examples/out/dbg_gmfs_crop_pad

Batch (JSONL):
python run_my_fvc4.py \
  --model ./models/DeepPrint_TexMinu_512/best_model.pyt \
  --data  ./data/fingerprints/FVC2000/Dbs/Db1_a \
  --configs-jsonl ./configs/roi_sweep.jsonl \
  --results-jsonl ./examples/out/roi_sweep_results.jsonl
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
# We use simple_api.fingerprint_segmentation
try:
    import pyfing.simple_api as pf
except Exception as e:
    pf = None


def imwrite_safe(path: str, img):
    """
    cv2.imwrite 只支持 1/3/4 通道。
    这里把各种中间格式强制转换成可保存的 uint8 图。
    """
    if img is None:
        return False

    # torch -> numpy
    try:
        import torch
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
    except Exception:
        pass

    img = np.asarray(img)

    # squeeze 掉多余维度
    while img.ndim > 3:
        img = np.squeeze(img, axis=0)

    # 如果是 (H,W,2) 这种，转成 3 通道方便保存
    if img.ndim == 3 and img.shape[2] not in (1, 3, 4):
        # 2通道最常见：把两张图叠一起了。这里取第一通道保存最直观
        img = img[:, :, 0]

    # (H,W,1) -> (H,W)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    # dtype 处理
    if img.dtype != np.uint8:
        # float/其他类型都压到 0~255
        mn, mx = float(np.min(img)), float(np.max(img))
        if mx - mn < 1e-12:
            img = np.zeros_like(img, dtype=np.uint8)
        else:
            img = ((img - mn) * 255.0 / (mx - mn)).clip(0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return cv2.imwrite(path, img)

def to_gray2d_uint8(img: np.ndarray) -> np.ndarray:
    """
    强制把任意输入整理成 2D 灰度 uint8 (H,W)。
    - 支持 HWC / CHW / (H,W,1)/(H,W,2) / torch tensor
    """
    if img is None:
        raise ValueError("to_gray2d_uint8: img is None")

    # torch -> numpy
    try:
        import torch
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
    except Exception:
        pass

    img = np.asarray(img)

    # squeeze batch-like dims
    while img.ndim > 3:
        img = np.squeeze(img, axis=0)

    if img.ndim == 3:
        # CHW?
        if img.shape[0] in (1, 2, 3, 4) and img.shape[1] > 8 and img.shape[2] > 8:
            # assume CHW
            c, h, w = img.shape
            if c == 1:
                img = img[0]
            elif c >= 3:
                # take first 3 channels as BGR-like and convert
                chw = img[:3]
                hwc = np.transpose(chw, (1, 2, 0))
                img = cv2.cvtColor(hwc.astype(np.uint8), cv2.COLOR_BGR2GRAY) if hwc.dtype == np.uint8 else cv2.cvtColor(
                    np.clip(hwc, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
                )
            else:
                # 2 channels -> take channel 0
                img = img[0]
        else:
            # HWC
            if img.shape[2] == 1:
                img = img[:, :, 0]
            elif img.shape[2] == 2:
                img = img[:, :, 0]
            else:
                img = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    if img.ndim != 2:
        raise ValueError(f"to_gray2d_uint8: expect 2D after conversion, got shape={img.shape}")

    if img.dtype != np.uint8:
        mn, mx = float(np.min(img)), float(np.max(img))
        if mx - mn < 1e-12:
            img = np.zeros_like(img, dtype=np.uint8)
        else:
            img = ((img - mn) * 255.0 / (mx - mn)).clip(0, 255).astype(np.uint8)

    return np.ascontiguousarray(img)

def crop_square_by_bbox(img2d: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """
    给定 bbox，围绕 bbox 中心裁剪出最大可能的正方形区域（在图内）。
    img2d: (H,W) uint8
    bbox: [x0,x1), [y0,y1)
    """
    h, w = img2d.shape
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    side = max(bw, bh)

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    sx0 = int(cx - side // 2)
    sy0 = int(cy - side // 2)
    sx1 = sx0 + side
    sy1 = sy0 + side

    # shift into image bounds
    if sx0 < 0:
        sx1 -= sx0
        sx0 = 0
    if sy0 < 0:
        sy1 -= sy0
        sy0 = 0
    if sx1 > w:
        d = sx1 - w
        sx0 = max(0, sx0 - d)
        sx1 = w
    if sy1 > h:
        d = sy1 - h
        sy0 = max(0, sy0 - d)
        sy1 = h

    # final clamp
    sx0, sy0 = max(0, sx0), max(0, sy0)
    sx1, sy1 = min(w, sx1), min(h, sy1)

    crop = img2d[sy0:sy1, sx0:sx1]
    return crop


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

def normalize_cfg(raw: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(defaults)
    cfg.update(raw)

    # normalize types (only if present)
    if "crop_sizes" in cfg:
        cfg["crop_sizes"] = [int(x) for x in cfg["crop_sizes"]]
    if "angles" in cfg:
        cfg["angles"] = [float(x) for x in cfg["angles"]]

    if "clahe_grid" in cfg:
        cfg["clahe_grid"] = [int(cfg["clahe_grid"][0]), int(cfg["clahe_grid"][1])]

    for k in ["blur_ksize", "rot_border", "bbox_margin", "mask_morph_ksize", "mask_morph_iter", "dpi"]:
        if k in cfg and cfg[k] is not None:
            cfg[k] = int(cfg[k])

    for k in ["fill", "clahe_clip", "alpha", "tex_delta", "minu_gap_lam"]:
        if k in cfg and cfg[k] is not None:
            cfg[k] = float(cfg[k])

    for k in ["tex_pair_topk"]:
        if k in cfg and cfg[k] is not None:
            cfg[k] = int(cfg[k])

    for k in ["tex_spike_clip_enable"]:
        if k in cfg and cfg[k] is not None:
            cfg[k] = bool(cfg[k])

    cfg["name"] = str(cfg.get("name", "unnamed"))
    cfg["roi_mode"] = str(cfg.get("roi_mode", "NONE"))
    cfg["mask_apply_mode"] = str(cfg.get("mask_apply_mode", "none"))
    cfg["mask_morph"] = str(cfg.get("mask_morph", "none"))

    return cfg


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
    mask_apply_mode: str = "none"       # none | white | black | crop_pad | bbox_crop
    dpi: int = 500

    # morphology
    mask_morph: str = "none"            # none | dilate | erode | open | close
    mask_morph_ksize: int = 0
    mask_morph_iter: int = 1

    # bbox crop
    bbox_margin: int = 0

    # debug
    debug_save_dir: Optional[str] = None


# ============================================================
# ROI helpers
# ============================================================

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def apply_mask_morph(mask: np.ndarray, morph: str, ksize: int, iters: int) -> np.ndarray:
    """
    mask: uint8 0/255
    """
    if morph is None:
        return mask
    morph = morph.lower().strip()
    if morph == "none" or ksize <= 0:
        return mask

    k = int(ksize)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    it = max(int(iters), 1)

    if morph == "dilate":
        return cv2.dilate(mask, kernel, iterations=it)
    if morph == "erode":
        return cv2.erode(mask, kernel, iterations=it)
    if morph == "open":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=it)
    if morph == "close":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=it)

    raise ValueError(f"Unknown mask morph: {morph}")

def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    returns (x0, y0, x1, y1) in pixel coords, inclusive-exclusive.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1

def bbox_expand(x0: int, y0: int, x1: int, y1: int, margin: int, w: int, h: int) -> Tuple[int, int, int, int]:
    m = int(margin)
    x0n = max(0, x0 - m)
    y0n = max(0, y0 - m)
    x1n = min(w, x1 + m)
    y1n = min(h, y1 + m)
    return x0n, y0n, x1n, y1n

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

    # normalize to 0/255 uint8
    m = np.asarray(m)
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8) * 255
    else:
        # some masks may be 0/1 or 0/255
        if m.max() <= 1:
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
    if mode == "white":
        out = img.copy()
        out[mask == 0] = 255
        return out
    if mode == "black":
        out = img.copy()
        out[mask == 0] = 0
        return out
    raise ValueError(f"apply_mask_to_image expects none/white/black, got: {mode}")


# ============================================================
# Loader (core)
# ============================================================

def make_loader(cfg: PreprocCfg):
    """
    This loader:
    1) read grayscale
    2) CLAHE
    3) rotate
    4) ROI/mask (optional, before cropping)
    5) choose crop method:
       - legacy crop by OTSU centroid (if mask_apply_mode in none/white/black)
       - crop by mask bbox (crop_pad/bbox_crop)
    6) pad_and_resize_to_deepprint_input_size
    """
    @staticmethod
    def _load(filepath: str):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {filepath}")
        img = _ensure_uint8(img)

        base = os.path.splitext(os.path.basename(filepath))[0]

        # 1) CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=float(cfg.clahe_clip),
            tileGridSize=(int(cfg.clahe_grid_x), int(cfg.clahe_grid_y)),
        )
        img_c = clahe.apply(img)

        # 2) Rotate
        if float(cfg.angle) != 0.0:
            h, w = img_c.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), float(cfg.angle), 1.0)
            img_r = cv2.warpAffine(img_c, M, (w, h), borderValue=int(cfg.rot_border))
        else:
            img_r = img_c

        # 3) ROI/mask
        mask = None
        roi_mode = (cfg.roi_mode or "NONE").upper().strip()
        if roi_mode != "NONE":
            mask = get_pyfing_mask(img_r, dpi=int(cfg.dpi), roi_mode=roi_mode)
            mask = apply_mask_morph(mask, cfg.mask_morph, cfg.mask_morph_ksize, cfg.mask_morph_iter)

        # debug dir
        dbg = cfg.debug_save_dir
        if dbg:
            os.makedirs(dbg, exist_ok=True)
            # save raw steps (safe)
            imwrite_safe(os.path.join(dbg, f"{base}_0_raw.png"), img)
            imwrite_safe(os.path.join(dbg, f"{base}_1_clahe.png"), img_c)
            imwrite_safe(os.path.join(dbg, f"{base}_2_rot.png"), img_r)
            if mask is not None:
                imwrite_safe(os.path.join(dbg, f"{base}_3_mask.png"), mask)

        # 4) choose crop strategy
        mode = (cfg.mask_apply_mode or "none").lower().strip()

        # A) bbox-crop mode (recommended): crop by mask bbox then pad/resize
        if mode in ("crop_pad", "bbox_crop"):
            if mask is None:
                mode = "none"
            else:
                bb = mask_to_bbox(mask)
                if bb is None:
                    mode = "none"
                else:
                    h, w = img_r.shape
                    x0, y0, x1, y1 = bbox_expand(*bb, margin=int(cfg.bbox_margin), w=w, h=h)

                    # 关键：确保 img_r 是 2D uint8
                    img_r2 = to_gray2d_uint8(img_r)

                    # 关键：正方形 crop（避免奇怪形状，且不会触发 flx 里维度判断问题）
                    img_crop = crop_square_by_bbox(img_r2, x0, y0, x1, y1)
                    img_crop = to_gray2d_uint8(img_crop)  # 再保险一次

                    if dbg:
                        vis = cv2.cvtColor(img_r2, cv2.COLOR_GRAY2BGR)
                        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
                        imwrite_safe(os.path.join(dbg, f"{base}_4_bbox_vis.png"), vis)
                        imwrite_safe(os.path.join(dbg, f"{base}_5_bbox_crop.png"), img_crop)

                    out = pad_and_resize_to_deepprint_input_size(img_crop, fill=float(cfg.fill))

                    if dbg:
                        imwrite_safe(os.path.join(dbg, f"{base}_6_final.png"), out)

                    return out


        # B) non-bbox modes: optionally apply mask as white/black, then legacy centroid crop
        if mask is not None and mode in ("white", "black"):
            img_m = apply_mask_to_image(img_r, mask, mode=mode)
        else:
            img_m = img_r

        # legacy crop by OTSU centroid (your original logic)
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

        if dbg:
            imwrite_safe(os.path.join(dbg, f"{base}_4_mask_applied.png"), img_m)
            vis = cv2.cvtColor(img_m, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis, (sx, sy), (sx + cs - 1, sy + cs - 1), (0, 255, 0), 2)
            imwrite_safe(os.path.join(dbg, f"{base}_5_legacy_crop_vis.png"), vis)
            imwrite_safe(os.path.join(dbg, f"{base}_6_legacy_crop.png"), img_crop)
            
        img_crop = to_gray2d_uint8(img_crop)
        out = pad_and_resize_to_deepprint_input_size(img_crop, fill=float(cfg.fill))
        if dbg:
            out_u8 = out
            if not isinstance(out_u8, np.ndarray):
                out_u8 = np.asarray(out_u8)
            if out_u8.dtype != np.uint8:
                out_u8 = np.clip(out_u8 * 255.0, 0, 255).astype(np.uint8) if out_u8.max() <= 1.5 else np.clip(out_u8, 0, 255).astype(np.uint8)
            imwrite_safe(os.path.join(dbg, f"{base}_7_final.png"), out_u8)

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
# view-pair robust score functions
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
                f"ROI={cfg.roi_mode} apply={cfg.mask_apply_mode} dpi={cfg.dpi} "
                f"morph={cfg.mask_morph} k={cfg.mask_morph_ksize} it={cfg.mask_morph_iter} margin={cfg.bbox_margin} | "
                f"CLAHE clip={cfg.clahe_clip} grid=({cfg.clahe_grid_x},{cfg.clahe_grid_y}) "
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
# CLI parsing
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

    # base preproc
    ap.add_argument("--clahe-clip", type=float, default=3.0)
    ap.add_argument("--clahe-grid", default="8,8")
    ap.add_argument("--blur-ksize", type=int, default=5)
    ap.add_argument("--rot-border", type=int, default=255)
    ap.add_argument("--fill", type=float, default=1.0)

    # ROI / mask
    ap.add_argument("--roi-mode", default="NONE",
                    choices=["NONE", "PYFING_GMFS", "PYFING_SUFS"])
    ap.add_argument("--mask-apply-mode", default="none",
                    choices=["none", "white", "black", "crop_pad", "bbox_crop"])
    ap.add_argument("--dpi", type=int, default=500)

    ap.add_argument("--mask-morph", default="none",
                    choices=["none", "dilate", "erode", "open", "close"])
    ap.add_argument("--mask-morph-ksize", type=int, default=0)
    ap.add_argument("--mask-morph-iter", type=int, default=1)
    ap.add_argument("--bbox-margin", type=int, default=0)

    ap.add_argument("--debug-save-dir", default=None)

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

    # batch
    ap.add_argument("--configs-jsonl", default=None)
    ap.add_argument("--results-jsonl", default="./examples/out/deepprint_results.jsonl")

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

    # load model once
    extractor = load_model(args.model, device=device, n_classes=8000, emb_dim=256)

    # default config (for batch filling)
    default_cfg = {
        "name": "cli_default",
        "crop_sizes": crop_sizes,
        "angles": angles,
        "clahe_clip": float(args.clahe_clip),
        "clahe_grid": [int(gx), int(gy)],
        "blur_ksize": int(args.blur_ksize),
        "rot_border": int(args.rot_border),
        "fill": float(args.fill),

        "roi_mode": args.roi_mode,
        "mask_apply_mode": args.mask_apply_mode,
        "dpi": int(args.dpi),
        "mask_morph": args.mask_morph,
        "mask_morph_ksize": int(args.mask_morph_ksize),
        "mask_morph_iter": int(args.mask_morph_iter),
        "bbox_margin": int(args.bbox_margin),

        "alpha": float(args.alpha),
        "tex_pair_topk": int(args.tex_pair_topk),
        "tex_spike_clip_enable": bool(tex_spike_clip_enable),
        "tex_delta": float(args.tex_delta),
        "minu_gap_lam": float(args.minu_gap_lam),

        "debug_save_dir": args.debug_save_dir,
    }

    # batch mode
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
                dpi=cfg.get("dpi", 500),
                mask_morph=cfg.get("mask_morph", "none"),
                mask_morph_ksize=cfg.get("mask_morph_ksize", 0),
                mask_morph_iter=cfg.get("mask_morph_iter", 1),
                bbox_margin=cfg.get("bbox_margin", 0),
                debug_save_dir=cfg.get("debug_save_dir", None),
            )

            t0 = time.time()
            tex_views, minu_views = extract_views_tex_minu(
                extractor=extractor,
                data_dir=args.data,
                crop_sizes=cfg["crop_sizes"],
                angles=cfg["angles"],
                preproc_base=pp,
                device=device,
            )
            print(f"[Info] Extracted views: V={tex_views.shape[1]} (crop_sizes={len(cfg['crop_sizes'])} x angles={len(cfg['angles'])})")

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

    # single run mode
    base_preproc = PreprocCfg(
        clahe_clip=float(args.clahe_clip),
        clahe_grid_x=int(gx),
        clahe_grid_y=int(gy),
        blur_ksize=int(args.blur_ksize),
        rot_border=int(args.rot_border),
        fill=float(args.fill),

        roi_mode=args.roi_mode,
        mask_apply_mode=args.mask_apply_mode,
        dpi=int(args.dpi),
        mask_morph=args.mask_morph,
        mask_morph_ksize=int(args.mask_morph_ksize),
        mask_morph_iter=int(args.mask_morph_iter),
        bbox_margin=int(args.bbox_margin),
        debug_save_dir=args.debug_save_dir,
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
