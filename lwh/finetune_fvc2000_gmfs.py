# -*- coding: utf-8 -*-
"""
finetune_fvc2000_gmfs.py

Fine-tune DeepPrint Tex+Minu on FVC2000-style folders with optional GMFS ROI.
Goal: improve 1:1 matching metrics (EER/FNMR) by avoiding label design mistakes.

Key design:
- Use finger_id (1..100) as class label (100-way), NOT (db,finger) (340-way).
- Use GMFS ROI preprocessing (optional) consistent with run_my_fvc4.
- Robust loader: after any crop, pad to square before flx pad_and_resize_to_deepprint_input_size.
- Train objective: CE on texture_logits + CE on minutia_logits (no minutia-map supervision).
- Save best_model_finetuned.pyt (by val_acc) and last_model_finetuned.pyt.

Notes:
- Training is fast because it trains per-image classification, not full 1:1 eval.
  Real matching quality must be evaluated using run_my_fvc4.py.
"""

import os
import sys
import argparse
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Make flx importable from project root
sys.path.append(os.getcwd())

from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size

# pyfing segmentation (optional)
try:
    import pyfing.simple_api as pf
except Exception:
    pf = None


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_u8(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def to_gray2d_uint8(img: np.ndarray) -> np.ndarray:
    """Force to 2D grayscale uint8 (H,W)."""
    img = np.asarray(img)
    while img.ndim > 3:
        img = np.squeeze(img, axis=0)

    if img.ndim == 3:
        # CHW?
        if img.shape[0] in (1, 2, 3, 4) and img.shape[1] > 8 and img.shape[2] > 8:
            c, h, w = img.shape
            if c == 1:
                img = img[0]
            else:
                img = img[0]
        else:
            # HWC
            if img.shape[2] >= 1:
                img = img[:, :, 0]

    if img.ndim != 2:
        raise ValueError(f"Expected 2D gray image, got shape={img.shape}")

    if img.dtype != np.uint8:
        mn, mx = float(np.min(img)), float(np.max(img))
        if mx - mn < 1e-12:
            img = np.zeros_like(img, dtype=np.uint8)
        else:
            img = ((img - mn) * 255.0 / (mx - mn)).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def pad_to_square_u8(img2d: np.ndarray, fill_u8: int = 255) -> np.ndarray:
    """Pad (H,W) to (S,S) with constant fill."""
    img2d = to_gray2d_uint8(img2d)
    h, w = img2d.shape
    if h == w:
        return img2d
    s = max(h, w)
    top = (s - h) // 2
    bottom = s - h - top
    left = (s - w) // 2
    right = s - w - left
    return cv2.copyMakeBorder(
        img2d, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=int(fill_u8)
    )


def apply_mask_morph(mask: np.ndarray, morph: str, ksize: int, iters: int) -> np.ndarray:
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

    raise ValueError(f"Unknown morph: {morph}")


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def bbox_expand(x0: int, y0: int, x1: int, y1: int, margin: int, w: int, h: int):
    m = int(margin)
    x0n = max(0, x0 - m)
    y0n = max(0, y0 - m)
    x1n = min(w, x1 + m)
    y1n = min(h, y1 + m)
    return x0n, y0n, x1n, y1n


def crop_square_by_bbox(img2d: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Crop a square around bbox center; clamp to image bounds."""
    img2d = to_gray2d_uint8(img2d)
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

    # shift into bounds
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

    sx0, sy0 = max(0, sx0), max(0, sy0)
    sx1, sy1 = min(w, sx1), min(h, sy1)

    crop = img2d[sy0:sy1, sx0:sx1]
    return crop


def get_pyfing_mask(img_u8: np.ndarray, dpi: int, roi_mode: str) -> np.ndarray:
    if pf is None:
        raise RuntimeError("pyfing.simple_api import failed. Cannot use PYFING_GMFS/SUFS.")
    roi_mode = roi_mode.upper().strip()
    if roi_mode == "PYFING_GMFS":
        m = pf.fingerprint_segmentation(img_u8, dpi=dpi, method="GMFS")
    elif roi_mode == "PYFING_SUFS":
        m = pf.fingerprint_segmentation(img_u8, dpi=dpi, method="SUFS")
    else:
        raise ValueError(f"Unsupported roi_mode: {roi_mode}")
    m = np.asarray(m)
    m = (m > 0).astype(np.uint8) * 255
    return m


def apply_mask_to_image(img_u8: np.ndarray, mask_u8: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower().strip()
    if mode == "none":
        return img_u8
    out = img_u8.copy()
    if mode == "white":
        out[mask_u8 == 0] = 255
        return out
    if mode == "black":
        out[mask_u8 == 0] = 0
        return out
    raise ValueError(f"mask apply mode must be none/white/black, got {mode}")


# -----------------------------
# Config
# -----------------------------
@dataclass
class PreprocCfg:
    clahe_clip: float = 3.0
    clahe_grid_x: int = 8
    clahe_grid_y: int = 8
    blur_ksize: int = 5
    rot_border: int = 255
    fill: float = 1.0

    # ROI / mask
    roi_mode: str = "NONE"              # NONE | PYFING_GMFS | PYFING_SUFS
    mask_apply_mode: str = "none"       # none | white | black | crop_pad | bbox_crop
    dpi: int = 500
    mask_morph: str = "none"            # none | dilate | erode | open | close
    mask_morph_ksize: int = 0
    mask_morph_iter: int = 1
    bbox_margin: int = 0


# -----------------------------
# Dataset
# -----------------------------
def parse_fvc_filename(name: str) -> Optional[Tuple[int, int]]:
    """
    FVC filenames: <finger>_<impression>.tif
    e.g. 1_2.tif -> finger=1, imp=2
    """
    base = os.path.splitext(os.path.basename(name))[0]
    if "_" not in base:
        return None
    a, b = base.split("_", 1)
    try:
        finger = int(a)
        imp = int(b)
        return finger, imp
    except Exception:
        return None


def list_fvc_images(db_dir: str) -> List[Tuple[str, str, int, int]]:
    """
    returns list of (path, db_name, finger, imp)
    """
    out = []
    db_name = os.path.basename(os.path.normpath(db_dir))
    for fn in sorted(os.listdir(db_dir)):
        if not fn.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")):
            continue
        parsed = parse_fvc_filename(fn)
        if parsed is None:
            continue
        finger, imp = parsed
        out.append((os.path.join(db_dir, fn), db_name, finger, imp))
    return out


class FVC2000FingerDataset(Dataset):
    """
    Training label: finger_id (1..100) -> 0..99
    Split: val_imp impression per db per finger goes to val set.
    """

    def __init__(
        self,
        items: List[Tuple[str, str, int, int]],
        preproc: PreprocCfg,
        crop_sizes: List[int],
        angles: List[float],
        is_train: bool,
        val_imp: int = 8,
    ):
        self.items = items
        self.preproc = preproc
        self.crop_sizes = list(crop_sizes)
        self.angles = list(angles)
        self.is_train = is_train
        self.val_imp = int(val_imp)

        # filter items based on is_train
        filtered = []
        for path, db, finger, imp in self.items:
            if self.is_train:
                if imp == self.val_imp:
                    continue
            else:
                if imp != self.val_imp:
                    continue
            filtered.append((path, db, finger, imp))
        self.items = filtered

        # class map: finger -> finger-1
        fingers = sorted(set(f for _, _, f, _ in self.items))
        self.class_map = {f: (f - 1) for f in fingers}

    def __len__(self):
        return len(self.items)

    def _load_one(self, path: str, crop_size: int, angle: float) -> torch.Tensor:
        cfg = self.preproc
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = ensure_u8(img)

        # CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=float(cfg.clahe_clip),
            tileGridSize=(int(cfg.clahe_grid_x), int(cfg.clahe_grid_y)),
        )
        img = clahe.apply(img)

        # rotate
        if float(angle) != 0.0:
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), float(angle), 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=int(cfg.rot_border))

        # ROI
        mask = None
        roi_mode = (cfg.roi_mode or "NONE").upper().strip()
        if roi_mode != "NONE":
            mask = get_pyfing_mask(img, dpi=int(cfg.dpi), roi_mode=roi_mode)
            mask = apply_mask_morph(mask, cfg.mask_morph, cfg.mask_morph_ksize, cfg.mask_morph_iter)

        mode = (cfg.mask_apply_mode or "none").lower().strip()

        fill_u8 = int(round(float(cfg.fill) * 255.0))

        # bbox crop mode
        if mode in ("crop_pad", "bbox_crop") and mask is not None:
            bb = mask_to_bbox(mask)
            if bb is not None:
                h, w = img.shape
                x0, y0, x1, y1 = bbox_expand(*bb, margin=int(cfg.bbox_margin), w=w, h=h)
                img_crop = crop_square_by_bbox(img, x0, y0, x1, y1)
                img_crop = pad_to_square_u8(img_crop, fill_u8=fill_u8)
                out = pad_and_resize_to_deepprint_input_size(img_crop, fill=float(cfg.fill))
                # out is usually CHW float in [0,1], but be robust
                out = np.asarray(out)
                if out.ndim == 2:
                    out = out[None, :, :]
                x = torch.from_numpy(out).float()
                return x

        # otherwise apply mask (white/black) then OTSU centroid crop
        if mask is not None and mode in ("white", "black"):
            img2 = apply_mask_to_image(img, mask, mode=mode)
        else:
            img2 = img

        # OTSU centroid crop to square crop_size
        k = int(cfg.blur_ksize)
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur(img2, (k, k), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.argwhere(th > 0)

        cy, cx = img2.shape[0] // 2, img2.shape[1] // 2
        if len(coords) > 0:
            cy, cx = coords.mean(0).astype(int)

        cs = int(crop_size)
        cs = min(cs, img2.shape[0], img2.shape[1])
        sy = max(0, min(cy - cs // 2, img2.shape[0] - cs))
        sx = max(0, min(cx - cs // 2, img2.shape[1] - cs))
        img_crop = img2[sy:sy + cs, sx:sx + cs]

        img_crop = pad_to_square_u8(img_crop, fill_u8=fill_u8)
        out = pad_and_resize_to_deepprint_input_size(img_crop, fill=float(cfg.fill))
        out = np.asarray(out)
        if out.ndim == 2:
            out = out[None, :, :]
        x = torch.from_numpy(out).float()
        return x

    def __getitem__(self, idx: int):
        path, db, finger, imp = self.items[idx]
        # random view for training
        if self.is_train:
            crop_size = random.choice(self.crop_sizes)
            angle = random.choice(self.angles)
        else:
            # deterministic for val: use middle value
            crop_size = self.crop_sizes[len(self.crop_sizes)//2]
            angle = self.angles[len(self.angles)//2]

        x = self._load_one(path, crop_size=crop_size, angle=angle)
        y = int(self.class_map[finger])  # 0..99
        return x, y, path, db, finger, imp


# -----------------------------
# Model load / trainable control
# -----------------------------
def load_model(model_path: str, device: str, n_classes: int = 8000, emb_dim: int = 256):
    extractor = get_DeepPrint_TexMinu(n_classes, emb_dim)
    ckpt = torch.load(model_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    md = extractor.model.state_dict()
    md.update({k: v for k, v in sd.items() if k in md and hasattr(v, "shape") and v.shape == md[k].shape})
    extractor.model.load_state_dict(md)

    extractor.model.to(device)
    return extractor.model


def set_trainable(model: nn.Module, mode: str):
    mode = mode.lower().strip()

    # freeze all first
    for p in model.parameters():
        p.requires_grad = False

    if mode == "head":
        # train only logits layers
        for name, m in model.named_modules():
            if name.endswith("texture_logits") or name.endswith("minutia_logits"):
                for p in m.parameters():
                    p.requires_grad = True
        return

    if mode == "last":
        # train embeddings + logits
        for name, m in model.named_modules():
            if (
                name.endswith("texture_branch")
                or name.endswith("minutia_embedding")
                or name.endswith("texture_logits")
                or name.endswith("minutia_logits")
            ):
                for p in m.parameters():
                    p.requires_grad = True
        return

    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    raise ValueError("trainable must be head/last/all")


# -----------------------------
# Training
# -----------------------------
def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return float((pred == y).float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-dbs", nargs="+", required=True)
    ap.add_argument("--outdir", default="./examples/out/finetune_fvc2000_gmfs")

    # views
    ap.add_argument("--crop-sizes", default="340,370,400,430")
    ap.add_argument("--angles", default="-15,0,15")

    # base preproc
    ap.add_argument("--clahe-clip", type=float, default=3.0)
    ap.add_argument("--clahe-grid", default="8,8")
    ap.add_argument("--blur-ksize", type=int, default=5)
    ap.add_argument("--rot-border", type=int, default=255)
    ap.add_argument("--fill", type=float, default=1.0)

    # ROI / mask
    ap.add_argument("--roi-mode", default="NONE", choices=["NONE", "PYFING_GMFS", "PYFING_SUFS"])
    ap.add_argument("--mask-apply-mode", default="none", choices=["none", "white", "black", "crop_pad", "bbox_crop"])
    ap.add_argument("--dpi", type=int, default=500)
    ap.add_argument("--mask-morph", default="none", choices=["none", "dilate", "erode", "open", "close"])
    ap.add_argument("--mask-morph-ksize", type=int, default=0)
    ap.add_argument("--mask-morph-iter", type=int, default=1)
    ap.add_argument("--bbox-margin", type=int, default=0)

    # model fusion/weight (used for training logits mixing)
    ap.add_argument("--alpha", type=float, default=5.0)

    # training
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--trainable", default="last", choices=["head", "last", "all"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--val-imp", type=int, default=8)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA not available, fallback to CPU.")
        device = "cpu"

    crop_sizes = [int(x) for x in args.crop_sizes.split(",") if x.strip()]
    angles = [float(x) for x in args.angles.split(",") if x.strip()]

    gx, gy = [int(x) for x in args.clahe_grid.split(",")]
    preproc = PreprocCfg(
        clahe_clip=float(args.clahe_clip),
        clahe_grid_x=int(gx),
        clahe_grid_y=int(gy),
        blur_ksize=int(args.blur_ksize),
        rot_border=int(args.rot_border),
        fill=float(args.fill),
        roi_mode=str(args.roi_mode),
        mask_apply_mode=str(args.mask_apply_mode),
        dpi=int(args.dpi),
        mask_morph=str(args.mask_morph),
        mask_morph_ksize=int(args.mask_morph_ksize),
        mask_morph_iter=int(args.mask_morph_iter),
        bbox_margin=int(args.bbox_margin),
    )

    # load all items
    items = []
    for db in args.train_dbs:
        items.extend(list_fvc_images(db))
    # Keep only fingers 1..100
    items = [it for it in items if 1 <= it[2] <= 100]

    # dataset sizes
    train_ds = FVC2000FingerDataset(items, preproc, crop_sizes, angles, is_train=True, val_imp=args.val_imp)
    val_ds = FVC2000FingerDataset(items, preproc, crop_sizes, angles, is_train=False, val_imp=args.val_imp)

    print(f"[Data] total images={len(items)} | classes(finger)=100")
    print(f"[Split] train={len(train_ds)} val={len(val_ds)} (val_imp={args.val_imp})")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(0, args.num_workers // 2), pin_memory=(device == "cuda"),
        drop_last=False
    )

    # model
    model = load_model(args.model, device=device, n_classes=8000, emb_dim=256)
    set_trainable(model, args.trainable)
    model.train()

    # We'll map logits (8000) -> 100 classes by a lightweight adapter,
    # because pretrained head is 8000-way.
    # Best practice: keep DeepPrint trunk, learn new 100-way heads on top of embeddings.
    # We take training output's texture_embeddings/minutia_embeddings and add new heads.
    tex_head = nn.Linear(256, 100).to(device)
    minu_head = nn.Linear(256, 100).to(device)

    # optimizer parameters
    params = []
    params += [p for p in model.parameters() if p.requires_grad]
    params += list(tex_head.parameters())
    params += list(minu_head.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    best_val = -1.0
    best_path = os.path.join(args.outdir, "best_model_finetuned.pyt")
    last_path = os.path.join(args.outdir, "last_model_finetuned.pyt")

    global_step = 0
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else 0

    def eval_val() -> float:
        model.eval()
        tex_head.eval()
        minu_head.eval()
        accs = []
        with torch.no_grad():
            for xb, yb, *_ in val_loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.long)
                out = model(xb)  # DeepPrintTrainingOutput in training, DeepPrintOutput in eval
                # In eval mode, it returns embeddings without logits; but model.eval() -> no_grad path.
                # Ensure we get embeddings:
                if hasattr(out, "texture_embeddings") and hasattr(out, "minutia_embeddings"):
                    tex_emb = out.texture_embeddings
                    minu_emb = out.minutia_embeddings
                else:
                    # DeepPrintOutput in eval returns (minutia_embeddings, texture_embeddings) in ctor order in dump,
                    # but safer to handle common tuple-like.
                    if isinstance(out, (tuple, list)) and len(out) >= 2:
                        minu_emb, tex_emb = out[0], out[1]
                    else:
                        raise RuntimeError("Unexpected model output in eval. Cannot get embeddings.")

                tex_emb = F.normalize(tex_emb, p=2, dim=1)
                minu_emb = F.normalize(minu_emb, p=2, dim=1)

                # fused logits (match eval idea: tex + alpha*minu)
                logits = tex_head(tex_emb) + float(args.alpha) * minu_head(minu_emb)
                accs.append(accuracy_top1(logits, yb))
        model.train()
        tex_head.train()
        minu_head.train()
        return float(np.mean(accs)) if accs else 0.0

    # warmup: sanity check one batch (helps catch loader errors early)
    xb0, yb0, *_ = next(iter(train_loader))
    print("[Sanity] one batch:", tuple(xb0.shape), tuple(yb0.shape))

    for epoch in range(1, args.epochs + 1):
        model.train()
        tex_head.train()
        minu_head.train()

        losses = []
        accs = []

        for xb, yb, *_ in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)

            # Training path: need embeddings -> set model.train() (we are in train)
            out = model(xb)
            if not (hasattr(out, "texture_embeddings") and hasattr(out, "minutia_embeddings")):
                raise RuntimeError("Unexpected model output in train. Need texture_embeddings/minutia_embeddings.")

            tex_emb = out.texture_embeddings
            minu_emb = out.minutia_embeddings

            tex_emb = F.normalize(tex_emb, p=2, dim=1)
            minu_emb = F.normalize(minu_emb, p=2, dim=1)

            logits = tex_head(tex_emb) + float(args.alpha) * minu_head(minu_emb)
            loss = ce(logits * float(args.scale), yb)

            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            accs.append(accuracy_top1(logits.detach(), yb.detach()))

            global_step += 1
            if max_steps > 0 and global_step >= max_steps:
                break

        val_acc = eval_val()
        tr_loss = float(np.mean(losses)) if losses else 0.0
        tr_acc = float(np.mean(accs)) if accs else 0.0

        print(f"[Epoch {epoch:03d}/{args.epochs}] loss={tr_loss:.4f} train_acc={tr_acc:.3f} val_acc={val_acc:.3f}")

        # save best
        if val_acc > best_val:
            best_val = val_acc
            ck = {
                "model_state_dict": model.state_dict(),
                "tex_head_state_dict": tex_head.state_dict(),
                "minu_head_state_dict": minu_head.state_dict(),
                "meta": {
                    "seed": args.seed,
                    "trainable": args.trainable,
                    "alpha": float(args.alpha),
                    "preproc": asdict(preproc),
                    "crop_sizes": crop_sizes,
                    "angles": angles,
                    "val_imp": int(args.val_imp),
                    "epoch": int(epoch),
                    "best_val_acc": float(best_val),
                },
            }
            torch.save(ck, best_path)
            print(f"[Saved] best -> {best_path} (val_acc={best_val:.3f})")

        if max_steps > 0 and global_step >= max_steps:
            print(f"[Stop] reached max_steps={max_steps}")
            break

    # save last
    ck = {
        "model_state_dict": model.state_dict(),
        "tex_head_state_dict": tex_head.state_dict(),
        "minu_head_state_dict": minu_head.state_dict(),
        "meta": {
            "seed": args.seed,
            "trainable": args.trainable,
            "alpha": float(args.alpha),
            "preproc": asdict(preproc),
            "crop_sizes": crop_sizes,
            "angles": angles,
            "val_imp": int(args.val_imp),
                    "epoch": int(epoch),
            "best_val_acc": float(best_val),
        },
    }
    torch.save(ck, last_path)
    print(f"[Saved] last -> {last_path}")


if __name__ == "__main__":
    main()
