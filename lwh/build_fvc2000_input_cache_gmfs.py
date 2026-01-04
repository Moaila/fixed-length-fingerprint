# -*- coding: utf-8 -*-
"""
build_fvc2000_input_cache_gmfs_v2.py

把 FVC2000 多个 DB 的图片预处理（GMFS + crop_pad + CLAHE 等）后，
缓存成 DeepPrint 输入（灰度 uint8 PNG），并写 meta.jsonl。

修复点：
- 无论中间张量/数组是 CHW/HWC/float，都强制变成 HxW uint8 再 imwrite
- meta 里写 cache_png 字段，供 cached finetune 直接读取
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2

import sys
sys.path.append(os.getcwd())

from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size

try:
    import pyfing.simple_api as pf
except Exception:
    pf = None


def ensure_gray_u8_2d(x) -> np.ndarray:
    """把任意输入转成 (H,W) uint8 灰度，保证 cv2.imwrite 可写。"""
    if x is None:
        raise ValueError("ensure_gray_u8_2d: input is None")

    # torch tensor -> numpy
    try:
        import torch
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
    except Exception:
        pass

    x = np.asarray(x)

    # squeeze 多余维度
    while x.ndim > 3:
        x = np.squeeze(x, axis=0)

    if x.ndim == 3:
        # CHW?
        if x.shape[0] in (1, 2, 3, 4) and x.shape[1] > 8 and x.shape[2] > 8:
            # CHW -> 取第0通道
            x = x[0]
        else:
            # HWC
            if x.shape[2] == 1:
                x = x[:, :, 0]
            elif x.shape[2] in (3, 4):
                x = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                # 其它通道数，取第0通道
                x = x[:, :, 0]

    if x.ndim != 2:
        raise ValueError(f"ensure_gray_u8_2d: expect 2D, got {x.shape}")

    # dtype -> uint8
    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        # 如果看起来是 0~1
        if x.max() <= 1.5:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(x)


def parse_fvc_name(path: Path) -> Tuple[str, int, int]:
    """
    从 /.../Db1_b/101_1.tif 解析：
    db = Db1_b
    finger = 101
    imp = 1
    """
    db = path.parent.name
    stem = path.stem  # "101_1"
    a, b = stem.split("_")
    return db, int(a), int(b)


def get_pyfing_mask(img_u8: np.ndarray, dpi: int, roi_mode: str) -> np.ndarray:
    if pf is None:
        raise RuntimeError("pyfing.simple_api import failed, cannot use PYFING_GMFS/SUFS.")
    roi_mode = roi_mode.upper().strip()
    if roi_mode == "PYFING_GMFS":
        m = pf.fingerprint_segmentation(img_u8, dpi=dpi, method="GMFS")
    elif roi_mode == "PYFING_SUFS":
        m = pf.fingerprint_segmentation(img_u8, dpi=dpi, method="SUFS")
    else:
        raise ValueError("roi_mode must be PYFING_GMFS or PYFING_SUFS")
    m = np.asarray(m)
    m = (m > 0).astype(np.uint8) * 255
    return m


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def bbox_expand(x0, y0, x1, y1, margin, w, h):
    m = int(margin)
    return max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)


def crop_square_by_bbox(img2d: np.ndarray, x0, y0, x1, y1) -> np.ndarray:
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

    if sx0 < 0:
        sx1 -= sx0; sx0 = 0
    if sy0 < 0:
        sy1 -= sy0; sy0 = 0
    if sx1 > w:
        d = sx1 - w
        sx0 = max(0, sx0 - d); sx1 = w
    if sy1 > h:
        d = sy1 - h
        sy0 = max(0, sy0 - d); sy1 = h

    return img2d[sy0:sy1, sx0:sx1]


def apply_mask_morph(mask: np.ndarray, morph: str, ksize: int, iters: int) -> np.ndarray:
    morph = (morph or "none").lower().strip()
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
    raise ValueError(f"Unknown morph: {morph}")


def build_one(src_path: Path, out_inputs: Path,
              roi_mode: str, mask_apply_mode: str, dpi: int,
              bbox_margin: int, mask_morph: str, mask_morph_ksize: int, mask_morph_iter: int,
              clahe_clip: float, clahe_grid: Tuple[int, int],
              fill: float = 1.0) -> Path:
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read: {src_path}")
    img = ensure_gray_u8_2d(img)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_grid[0]), int(clahe_grid[1])))
    img = clahe.apply(img)

    # ROI + crop_pad
    if roi_mode.upper() != "NONE":
        mask = get_pyfing_mask(img, dpi=dpi, roi_mode=roi_mode)
        mask = apply_mask_morph(mask, mask_morph, mask_morph_ksize, mask_morph_iter)
    else:
        mask = None

    mode = (mask_apply_mode or "none").lower().strip()
    if mode in ("crop_pad", "bbox_crop") and mask is not None:
        bb = mask_to_bbox(mask)
        if bb is not None:
            h, w = img.shape
            x0, y0, x1, y1 = bbox_expand(*bb, margin=bbox_margin, w=w, h=h)
            img_crop = crop_square_by_bbox(img, x0, y0, x1, y1)
        else:
            img_crop = img
    elif mode == "white" and mask is not None:
        img_crop = img.copy()
        img_crop[mask == 0] = 255
    elif mode == "black" and mask is not None:
        img_crop = img.copy()
        img_crop[mask == 0] = 0
    else:
        img_crop = img

    # DeepPrint input
    out = pad_and_resize_to_deepprint_input_size(img_crop, fill=float(fill))
    out_u8 = ensure_gray_u8_2d(out)

    # write png
    db, finger, imp = parse_fvc_name(src_path)
    name = f"{db}__{finger:03d}__{imp}.png"
    out_path = out_inputs / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), out_u8)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--roi-mode", default="PYFING_GMFS", choices=["NONE", "PYFING_GMFS", "PYFING_SUFS"])
    ap.add_argument("--mask-apply-mode", default="crop_pad", choices=["none", "white", "black", "crop_pad", "bbox_crop"])
    ap.add_argument("--dpi", type=int, default=500)
    ap.add_argument("--bbox-margin", type=int, default=20)
    ap.add_argument("--mask-morph", default="dilate", choices=["none", "dilate", "erode", "open", "close"])
    ap.add_argument("--mask-morph-ksize", type=int, default=5)
    ap.add_argument("--mask-morph-iter", type=int, default=1)

    ap.add_argument("--clahe-clip", type=float, default=3.0)
    ap.add_argument("--clahe-grid", default="8,8")
    ap.add_argument("--fill", type=float, default=1.0)

    args = ap.parse_args()
    outdir = Path(args.outdir)
    out_inputs = outdir / "inputs"
    meta_path = outdir / "meta.jsonl"
    out_inputs.mkdir(parents=True, exist_ok=True)

    gx, gy = [int(x.strip()) for x in args.clahe_grid.split(",")]

    # collect images
    img_paths = []
    for db in args.dbs:
        dbp = Path(db)
        img_paths += sorted(dbp.glob("*.tif"))
        img_paths += sorted(dbp.glob("*.png"))
        img_paths += sorted(dbp.glob("*.bmp"))
        img_paths += sorted(dbp.glob("*.jpg"))

    print(f"[Cache] total images = {len(img_paths)}")
    print(f"[Cache] outdir = {outdir}")

    # write meta
    if meta_path.exists():
        meta_path.unlink()

    ok_cnt = 0
    with meta_path.open("w", encoding="utf-8") as f:
        for p in tqdm(img_paths, desc="Building cache", ncols=100):
            rec = {"src_path": str(p)}
            try:
                cache_png = build_one(
                    src_path=p,
                    out_inputs=out_inputs,
                    roi_mode=args.roi_mode,
                    mask_apply_mode=args.mask_apply_mode,
                    dpi=args.dpi,
                    bbox_margin=args.bbox_margin,
                    mask_morph=args.mask_morph,
                    mask_morph_ksize=args.mask_morph_ksize,
                    mask_morph_iter=args.mask_morph_iter,
                    clahe_clip=args.clahe_clip,
                    clahe_grid=(gx, gy),
                    fill=args.fill,
                )
                db, finger, imp = parse_fvc_name(p)
                rec.update({
                    "ok": True,
                    "cache_png": str(cache_png),
                    "db": db,
                    "finger": finger,
                    "imp": imp,
                })
                ok_cnt += 1
            except Exception as e:
                rec.update({"ok": False, "error": str(e)})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[OK] cache built:")
    print("  inputs:", out_inputs)
    print("  meta:  ", meta_path)
    print("  ok_cnt:", ok_cnt)


if __name__ == "__main__":
    from tqdm import tqdm
    main()
