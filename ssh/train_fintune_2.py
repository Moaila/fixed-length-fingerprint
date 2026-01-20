# -*- coding: utf-8 -*-
r"""
finetune_fvc_metric.py

Fine-tune DeepPrint on combined FVC datasets (2002/2004 DB1-4)

Strategy:
- Train on ALL available DBs (DB1-4) to maximize class diversity (~800 identities).
- Use ArcFace to enforce compact intra-class variance.

"""

"""
python ./ssh/train_fintune_2.py --dirs "./ssh/data/fingerprints/FVC2002/Db1_a" "./ssh/data/fingerprints/FVC2002/Db2_a" "./ssh/data/fingerprints/FVC2002/Db3_a" "./ssh/data/fingerprints/FVC2002/Db4_a" "./ssh/data/fingerprints/FVC2004/Db1_A" "./ssh/data/fingerprints/FVC2004/Db2_A" "./ssh/data/fingerprints/FVC2004/Db3_A" "./ssh/data/fingerprints/FVC2004/Db4_A"   --batch_size 32  --alpha 1.0  --margin 0.5
"""

import os
import sys
import argparse
import random
import logging
import glob
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Sampler
from sklearn.metrics import roc_curve

# Make project importable
sys.path.append(os.getcwd())

from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu
# 确保你的 flx 包里有这个函数，如果没有请确保把之前的 padding 逻辑复制过来
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("finetune_fvc_metric.log", encoding="utf-8")]
)
logger = logging.getLogger("fvc_metric")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Utils: Image Processing
# -----------------------------
def to_gray2d_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure image is (H,W) uint8."""
    img = np.asarray(img)
    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img[0]
        elif img.shape[-1] == 1:
            img = img[:, :, 0]
        else:
            # aggressive conversion
            img = img[:, :, 0]
    
    if img.dtype != np.uint8:
        # Normalize to 0-255 if float
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def apply_fvc_aug(img_u8: np.ndarray, rot_deg: float = 15.0, trans_px: int = 15) -> np.ndarray:
    """
    Stronger augmentation for FVC (larger deformations).
    """
    h, w = img_u8.shape
    
    # Random Rotation
    angle = random.uniform(-rot_deg, rot_deg)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    # Fill with 255 (white background common in processed FVC) or 0? 
    # DeepPrint usually expects 1.0 (white) padding if normalized, check your preprocessing.
    # Assuming standard white background for fingerprints:
    img = cv2.warpAffine(img_u8, M, (w, h), borderValue=255)

    # Random Translation
    tx = random.randint(-trans_px, trans_px)
    ty = random.randint(-trans_px, trans_px)
    M2 = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M2, (w, h), borderValue=255)

    return img


# -----------------------------
# Dataset: Multi-DB FVC Loader
# -----------------------------
class FVCMultiDBDataset(TorchDataset):
    """
    Loads images from multiple FVC database directories.
    Handles ID shifting so 2002_DB1_Finger1 != 2004_DB1_Finger1.
    """
    def __init__(self, db_dirs: List[str], is_train: bool, augment: bool, val_imp_idx: int = 7):
        """
        Args:
            db_dirs: List of paths to folders containing .tif images (e.g. ['path/to/2002/DB1', 'path/to/2004/DB1'])
            is_train: if True, exclude val impressions.
            augment: apply geometric aug.
            val_imp_idx: 0-based index for validation. FVC usually has 8 imps (0-7). 
                         If val_imp_idx=7, then imp 7 is Val, 0-6 are Train.
        """
        self.is_train = is_train
        self.augment = augment
        self.val_imp_idx = val_imp_idx
        
        self.samples = [] # (path, global_label)
        self.images_ram = [] # Cache
        
        global_id_offset = 0
        
        valid_exts = {".tif", ".tiff", ".png", ".jpg", ".bmp"}

        logger.info(f"[{'Train' if is_train else 'Val'}] Scanning {len(db_dirs)} directories...")

        for db_dir in db_dirs:
            if not os.path.exists(db_dir):
                logger.warning(f"Directory not found, skipping: {db_dir}")
                continue
                
            # Scan files
            files = sorted([f for f in os.listdir(db_dir) if os.path.splitext(f)[1].lower() in valid_exts])
            
            # Identify max subject ID in this folder to update offset later
            max_sub_in_folder = 0
            count_added = 0
            
            for fn in files:
                # Expected format: "101_1.tif" or "1_1.tif"
                # Split logic provided by user
                base = os.path.splitext(fn)[0]
                if "_" not in base:
                    continue
                try:
                    parts = base.split("_")
                    subject_id = int(parts[0])      # 1-based usually
                    impression_id = int(parts[1])   # 1-based usually
                except ValueError:
                    continue

                # Convert to 0-based
                sub_idx = subject_id - 1
                imp_idx = impression_id - 1
                
                if sub_idx > max_sub_in_folder:
                    max_sub_in_folder = sub_idx
                
                # Split Train/Val
                # If is_train: keep if imp_idx != val_imp_idx
                # If val: keep if imp_idx == val_imp_idx
                if self.is_train:
                    if imp_idx == self.val_imp_idx:
                        continue
                else:
                    if imp_idx != self.val_imp_idx:
                        continue
                
                # Global Label
                global_label = global_id_offset + sub_idx
                full_path = os.path.join(db_dir, fn)
                
                self.samples.append((full_path, global_label))
                count_added += 1

            logger.info(f"  -> Loaded {count_added} images from {os.path.basename(db_dir)} (Offset: {global_id_offset})")
            
            # Update offset for next DB (e.g., if this DB went up to finger 100, next starts at 100)
            # FVC usually has 100-110 fingers.
            global_id_offset += (max_sub_in_folder + 1)

        # Re-map labels to be contiguous 0..N-1 (in case some DBs have gaps or few fingers)
        unique_labels = sorted(list(set(s[1] for s in self.samples)))
        self.label_map = {old: new for new, old in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        
        # Update samples with contiguous labels
        self.samples = [(p, self.label_map[l]) for p, l in self.samples]
        self.labels = [s[1] for s in self.samples] # For Sampler
        
        # Preload to RAM
        logger.info(f"[{'Train' if is_train else 'Val'}] Preloading {len(self.samples)} images to RAM...")
        self._preload()
        
    def _preload(self):
        for path, _ in self.samples:
            # 1. Read
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # create dummy if failed (should not happen)
                img = np.zeros((299, 299), dtype=np.uint8)
            
            # 2. Resize/Pad (Crucial: DeepPrint expects specific input size, e.g., 299x299 or what pad_and_resize does)
            # Using the helper function provided by user context
            # Note: pad_and_resize usually returns a float tensor in [0,1] or numpy. 
            # We want to store uint8 in RAM to save space, then augment, then float.
            
            # Let's do raw resize to square logic here to keep RAM uint8, 
            # OR just trust the system RAM is enough (FVC is small).
            # Let's use the helper but convert back to uint8 for storage if needed. 
            # Actually, to be safe and use user's logic:
            
            # We will store the RAW image (or lightly processed) and resize in __getitem__ 
            # OR resize here. Resizing here is faster for training.
            
            # Logic: Pad to square -> Resize -> Store
            img = to_gray2d_uint8(img)
            
            # Simple pad to square logic to ensure aspect ratio is kept
            h, w = img.shape
            s = max(h, w)
            pad_img = np.full((s, s), 255, dtype=np.uint8)
            y_off = (s - h) // 2
            x_off = (s - w) // 2
            pad_img[y_off:y_off+h, x_off:x_off+w] = img
            
            # Resize to generic reasonable size (e.g. 320x320) to save RAM, 
            # final resize to DeepPrint input happens in __getitem__ via helper if needed,
            # BUT user's helper `pad_and_resize...` does it all.
            # Let's use `pad_and_resize` in __getitem__ to allow Augmentation on the raw-ish image.
            
            self.images_ram.append(pad_img)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_u8 = self.images_ram[idx].copy()
        label = self.samples[idx][1]
        
        # 1. Augmentation (on uint8)
        if self.is_train and self.augment:
            img_u8 = apply_fvc_aug(img_u8, rot_deg=10.0, trans_px=10)
            
        # 2. Final DeepPrint Preprocessing (Pad, Resize, Norm)
        # Assuming pad_and_resize_to_deepprint_input_size returns a Tensor [C,H,W]
        # If it's not available, we implement a fallback
        try:
            # We pass fill=1.0 (white) because FVC is usually white bg
            x = pad_and_resize_to_deepprint_input_size(img_u8, fill=1.0)
            
            # If helper returns numpy, convert to tensor
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            
            # Ensure shape (1, H, W)
            if x.ndim == 2:
                x = x.unsqueeze(0)
        except Exception as e:
            # Fallback if import failed
            x_fallback = cv2.resize(img_u8, (299, 299))
            x = torch.from_numpy(x_fallback).float() / 255.0
            x = x.unsqueeze(0)
            
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# -----------------------------
# Batch Sampler (P classes * K samples)
# -----------------------------
class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, labels: List[int], n_classes: int, n_samples: int):
        self.labels = np.array(labels)
        self.classes = sorted(list(set(labels)))
        
        self.cls_indices = {c: np.where(self.labels == c)[0].tolist() for c in self.classes}
        
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_classes * self.n_samples
        
        # Calculate roughly how many batches to cover data
        self.n_batches = len(labels) // self.batch_size
        if self.n_batches < 1: self.n_batches = 1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            # Pick P classes
            chosen_cls = random.sample(self.classes, self.n_classes) if len(self.classes) > self.n_classes else self.classes
            
            batch = []
            for c in chosen_cls:
                indices = self.cls_indices[c]
                # Pick K samples (with replacement if not enough)
                if len(indices) >= self.n_samples:
                    batch.extend(random.sample(indices, self.n_samples))
                else:
                    batch.extend(random.choices(indices, k=self.n_samples))
            
            yield batch


# -----------------------------
# Model Heads & Utils
# -----------------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

def extract_emb(out):
    # Robust extraction for DeepPrint structure
    if hasattr(out, "texture_embeddings"): return out.texture_embeddings, out.minutia_embeddings
    if isinstance(out, (tuple, list)): return out[1], out[0] # assuming (minu, tex) or vice versa, check your model!
    # Fallback assumption: Tex is usually [B, 256], Minu is [B, 256]
    return out[0], out[1] # Careful here

def fuse(tex, minu, alpha=2.0):
    # FVC quality varies, lower alpha (e.g. 2.0) puts more trust in Texture than Minutiae
    t = F.normalize(tex, p=2, dim=1)
    m = F.normalize(minu, p=2, dim=1)
    return torch.cat([t, alpha * m], dim=1)


# -----------------------------
# Main Training
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    # List of directories: e.g. --dirs /data/FVC2002/DB1 /data/FVC2004/DB1 ...
    parser.add_argument("--dirs", nargs='+', required=True, help="List of FVC DB directories")
    parser.add_argument("--pretrained", default="./ssh/example-model/best_model.pyt", help="Path to best_model.pyt")
    parser.add_argument("--outdir", default="./fvc_result")
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    
    # ArcFace params
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=64.0)
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for Minutiae (Lower than SD4's 5.0 is safer for FVC)")
    
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    
    # 1. Dataset
    # Train on Imp 0-6 (Imp 1-7 in filename), Val on Imp 7 (Imp 8 in filename)
    train_ds = FVCMultiDBDataset(args.dirs, is_train=True, augment=True, val_imp_idx=7)
    val_ds = FVCMultiDBDataset(args.dirs, is_train=False, augment=False, val_imp_idx=7)
    
    logger.info(f"Total Classes: {train_ds.num_classes}")
    
    # 2. Sampler (PK)
    # P classes, K=4 samples per class (Since FVC has ~7 training samples, K=4 is good)
    K = 4
    P = args.batch_size // K
    sampler = BalancedBatchSampler(train_ds.labels, n_classes=P, n_samples=K)
    
    train_loader = TorchDataLoader(train_ds, batch_sampler=sampler, num_workers=0) # Windows set workers=0
    val_loader = TorchDataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 3. Model
    # Note: Initialize with larger class count to avoid size mismatch if pretraining was large
    extractor = get_DeepPrint_TexMinu(num_training_subjects=8000, num_dims=256)
    
    # Load state dict
    ckpt = torch.load(args.pretrained, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = extractor.model
    # Loose load
    model_keys = model.state_dict().keys()
    sd = {k: v for k, v in sd.items() if k in model_keys and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(sd, strict=False)
    model.to(device)
    
    # 4. ArcFace Head
    fused_dim = 256 + 256 # Concat
    head = ArcMarginProduct(fused_dim, train_ds.num_classes, s=args.scale, m=args.margin).to(device)
    
    optimizer = optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Loop
    best_loss = 999.0
    
    for epoch in range(1, args.epochs+1):
        model.train()
        head.train()
        losses = []
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            
            out = model(xb)
            tex, minu = extract_emb(out)
            emb = fuse(tex, minu, alpha=args.alpha)

            #logger.info(f"tex {tex.shape} minu {minu.shape} emb {emb.shape} y {yb[:8].tolist()}")
            # Re-normalize before ArcFace? Usually fuse() does norm, but ArcFace does it internally too.
            # Our fuse() returns normalized cat, ArcFace normalizes input X. Double norm is fine.
            logits = head(emb, yb)
            loss = criterion(logits, yb)
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        avg_loss = np.mean(losses)
        logger.info(f"Epoch {epoch} Loss: {avg_loss:.4f}")
        
        # Save logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.outdir, "best_fvc_model.pyt"))

    # Save last
    torch.save(model.state_dict(), os.path.join(args.outdir, "last_fvc_model.pyt"))
    logger.info("Done.")

if __name__ == "__main__":
    main()