# -*- coding: utf-8 -*-
r"""
finetune_nist_sd4_metric.py

Fine-tune DeepPrint Tex+Minu on NIST SD4 (2000 subjects, 2 impressions each)
with metric-friendly objectives:
- ArcFace / CosFace (recommended)
- Batch-hard Triplet loss

"""

import os
import sys
import argparse
import random
import logging
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

# make project importable
sys.path.append(os.getcwd())

from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
from flx.data.image_loader import NistSD4Dataset
from flx.data.transformed_image_loader import TransformedImageLoader
from flx.image_processing.binarization import LazilyAllocatedBinarizer
from flx.data.dataset import Identifier, IdentifierSet, Dataset as FLXDataset


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("finetune_nist_sd4_metric.log", encoding="utf-8")]
)
logger = logging.getLogger("sd4_metric")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Utils: image handling
# -----------------------------
def to_gray2d_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3:
        # (1,H,W) or (H,W,1) -> (H,W)
        if img.shape[0] == 1:
            img = img[0]
        elif img.shape[-1] == 1:
            img = img[:, :, 0]
        else:
            img = img[:, :, 0]
    if img.ndim != 2:
        raise ValueError(f"Expected 2D gray image, got {img.shape}")

    if img.dtype != np.uint8:
        mn, mx = float(img.min()), float(img.max())
        if mx <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        elif mx > 255.0 or mn < 0.0:
            img = ((img - mn) * 255.0 / (mx - mn + 1e-6)).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def apply_basic_aug(img_u8: np.ndarray, rot_deg: float = 15.0, trans_px: int = 25) -> np.ndarray:
    """Aug similar to repo description: rotation + translation."""
    h, w = img_u8.shape

    angle = random.uniform(-rot_deg, rot_deg)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img_u8, M, (w, h), borderValue=255)

    tx = random.randint(-trans_px, trans_px)
    ty = random.randint(-trans_px, trans_px)
    M2 = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M2, (w, h), borderValue=255)

    return img


# -----------------------------
# Dataset (preload aligned by key)
# -----------------------------
class NISTSD4MetricDataset(TorchDataset):
    """
    Each subject has 2 impressions: (0=f, 1=s)
    Preload with dict keyed by (subject, impression) to avoid IdentifierSet reorder mismatch.

    shared_cache:
        reuse the already preloaded cache (avoid train+val preload twice)
    """

    def __init__(self, data_dir: str, subject_ids: List[int], is_train: bool, augment: bool,
                 shared_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = None):
        self.data_dir = data_dir
        self.subject_ids = list(subject_ids)
        self.is_train = bool(is_train)
        self.augment = bool(augment)

        sd4 = NistSD4Dataset(data_dir)
        self.transforms = [
            LazilyAllocatedBinarizer(4.8),
            pad_and_resize_to_deepprint_input_size,
        ]
        self.image_loader = TransformedImageLoader(images=sd4, poses=None, transforms=self.transforms)

        self.samples: List[Tuple[int, int]] = []
        for s in self.subject_ids:
            for imp in [0, 1]:
                self.samples.append((s, imp))

        uniq = sorted(set(self.subject_ids))
        self.subject_to_label = {s: i for i, s in enumerate(uniq)}
        self.num_classes = len(uniq)

        if shared_cache is not None:
            self.img_by_key = shared_cache
            logger.info(f"[Preload] reuse shared cache: {len(self.img_by_key)} images")
        else:
            self.img_by_key: Dict[Tuple[int, int], np.ndarray] = {}
            self._preload_images()

        self.labels = [self.subject_to_label[s] for (s, _imp) in self.samples]
        logger.info(f"[Dataset] samples={len(self.samples)} classes={self.num_classes} augment={self.augment}")

    def _preload_images(self):
        logger.info("[Preload] start ...")
        identifiers = [Identifier(subject=s, impression=imp) for (s, imp) in self.samples]

        id_set = IdentifierSet(identifiers)
        id_list = list(id_set)
        flx_ds = FLXDataset(self.image_loader, id_set)

        assert len(flx_ds) == len(id_list), "FLXDataset length mismatch with IdentifierSet order."

        for i in range(len(flx_ds)):
            img = flx_ds[i]
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()

            img_u8 = to_gray2d_uint8(img)
            key = (int(id_list[i].subject), int(id_list[i].impression))
            self.img_by_key[key] = img_u8

        miss = 0
        for s, imp in self.samples:
            if (s, imp) not in self.img_by_key:
                miss += 1
        if miss > 0:
            raise RuntimeError(f"[Preload] missing {miss} images after preload mapping!")

        logger.info(f"[Preload] done: {len(self.img_by_key)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s, imp = self.samples[idx]
        label = self.subject_to_label[s]

        img = self.img_by_key[(s, imp)].copy()

        if self.is_train and self.augment:
            img = apply_basic_aug(img)

        x = img.astype(np.float32) / 255.0
        x = x[None, :, :]
        x = torch.from_numpy(x).float()
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# -----------------------------
# Balanced batch sampler (P classes, K samples per class)
# SD4 has only 2 samples/class -> set K=2
# -----------------------------
class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, labels: List[int], n_classes: int, n_samples: int, batches_per_epoch: Optional[int] = None):
        self.labels = np.array(labels, dtype=np.int64)
        self.label_set = np.unique(self.labels).tolist()
        self.label_to_indices: Dict[int, List[int]] = {l: np.where(self.labels == l)[0].tolist() for l in self.label_set}

        for l in self.label_set:
            random.shuffle(self.label_to_indices[l])

        self.n_classes = int(n_classes)
        self.n_samples = int(n_samples)
        self.batch_size = self.n_classes * self.n_samples

        if batches_per_epoch is None:
            self.batches_per_epoch = max(1, len(self.labels) // self.batch_size)
        else:
            self.batches_per_epoch = int(batches_per_epoch)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            classes = random.sample(self.label_set, self.n_classes)
            batch = []
            for c in classes:
                idxs = self.label_to_indices[c]
                if len(idxs) < self.n_samples:
                    continue
                chosen = random.sample(idxs, self.n_samples)
                batch.extend(chosen)

            if len(batch) < self.batch_size:
                pool = np.arange(len(self.labels)).tolist()
                random.shuffle(pool)
                batch = (batch + pool)[:self.batch_size]

            yield batch


# -----------------------------
# DeepPrint embedding extraction (robust)
# -----------------------------
def extract_tex_minu_embeddings(out):
    if hasattr(out, "texture_embeddings") and hasattr(out, "minutia_embeddings"):
        return out.texture_embeddings, out.minutia_embeddings

    if isinstance(out, (tuple, list)) and len(out) >= 2:
        a, b = out[0], out[1]
        return b, a

    raise RuntimeError("Cannot extract embeddings from model output.")


def fuse_embedding(tex_emb, minu_emb, alpha: float = 5.0, mode: str = "concat"):
    tex = F.normalize(tex_emb, p=2, dim=1)
    minu = F.normalize(minu_emb, p=2, dim=1)

    if mode == "sum":
        emb = tex + float(alpha) * minu
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    if mode == "concat":
        emb = torch.cat([tex, float(alpha) * minu], dim=1)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    raise ValueError("fuse mode must be sum/concat")


# -----------------------------
# ArcFace / CosFace heads
# -----------------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.50, easy_margin: bool = False):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.s = float(s)
        self.m = float(m)
        self.easy_margin = bool(easy_margin)

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = np.cos(self.m)
        self.sin_m = np.sin(self.m)
        self.th = np.cos(np.pi - self.m)
        self.mm = np.sin(np.pi - self.m) * self.m

    def forward(self, x: torch.Tensor, label: torch.Tensor):
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(x, W).clamp(-1.0, 1.0)
        sine = torch.sqrt((1.0 - cosine * cosine).clamp_min(1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits


class CosFace(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.35):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.s = float(s)
        self.m = float(m)

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, label: torch.Tensor):
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(x, W).clamp(-1.0, 1.0)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        logits = cosine - one_hot * self.m
        logits *= self.s
        return logits


# -----------------------------
# Triplet loss (batch-hard)
# -----------------------------
class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = float(margin)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor):
        emb = F.normalize(emb, p=2, dim=1)
        dist = torch.cdist(emb, emb, p=2)  # [B,B]

        labels = labels.view(-1)
        mask_pos = labels.unsqueeze(1).eq(labels.unsqueeze(0))
        mask_pos.fill_diagonal_(False)

        dist_pos = dist * mask_pos.float()
        hardest_pos, _ = dist_pos.max(dim=1)

        dist_neg = dist + (1e5 * mask_pos.float())
        hardest_neg, _ = dist_neg.min(dim=1)

        loss = F.relu(hardest_pos - hardest_neg + self.margin).mean()
        return loss


def batch_recall_at_1(emb: torch.Tensor, labels: torch.Tensor) -> float:
    emb = F.normalize(emb, p=2, dim=1)
    dist = torch.cdist(emb, emb, p=2)
    dist.fill_diagonal_(1e9)
    nn_idx = torch.argmin(dist, dim=1)
    pred = labels[nn_idx]
    return float((pred == labels).float().mean().item())


# -----------------------------
# Model load
# -----------------------------
def load_backbone(pretrained_path: str, device: str, emb_dim: int = 256, pretrain_classes: int = 8000) -> nn.Module:
    model = get_DeepPrint_TexMinu(num_training_subjects=pretrain_classes, num_dims=emb_dim).model
    ckpt = torch.load(pretrained_path, map_location="cpu")

    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    md = model.state_dict()

    loadable = {}
    for k, v in sd.items():
        if k in md and hasattr(v, "shape") and v.shape == md[k].shape:
            loadable[k] = v
    md.update(loadable)
    model.load_state_dict(md, strict=False)

    model.to(device)
    return model


# -----------------------------
# Eval: EER
# -----------------------------
@torch.no_grad()
def eval_eer(model, loader, device: str, alpha: float, fuse_mode: str, impostor_pairs: int = 50000) -> float:
    model.eval()
    all_emb = []
    all_lab = []

    for xb, yb in loader:
        xb = xb.to(device=device, dtype=torch.float32)
        yb = yb.to(device=device, dtype=torch.long)

        out = model(xb)
        tex, minu = extract_tex_minu_embeddings(out)
        emb = fuse_embedding(tex, minu, alpha=alpha, mode=fuse_mode)

        all_emb.append(emb.cpu().numpy())
        all_lab.append(yb.cpu().numpy())

    E = np.concatenate(all_emb, axis=0)
    L = np.concatenate(all_lab, axis=0)

    label_to_idx = {}
    for i, lab in enumerate(L):
        label_to_idx.setdefault(int(lab), []).append(i)

    genuine = []
    for idxs in label_to_idx.values():
        if len(idxs) >= 2:
            i0, i1 = idxs[0], idxs[1]
            genuine.append(np.linalg.norm(E[i0] - E[i1]))

    if len(genuine) == 0:
        return 1.0

    genuine = np.array(genuine, dtype=np.float32)

    n = len(L)
    imp = []
    tries = 0
    while len(imp) < int(impostor_pairs) and tries < int(impostor_pairs) * 2:
        tries += 1
        i = random.randrange(n)
        j = random.randrange(n)
        if L[i] == L[j]:
            continue
        imp.append(np.linalg.norm(E[i] - E[j]))
    imp = np.array(imp, dtype=np.float32)

    scores = np.concatenate([-genuine, -imp])
    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(imp)])

    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    model.train()
    return eer


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str,
                    default=r"D:\AAAYan\ZhiWen\FixLength\fixed-length-fingerprint\ssh\data\fingerprints\nist_sd4")
    ap.add_argument("--pretrained_path", type=str,
                    default=r"D:\AAAYan\ZhiWen\FixLength\fixed-length-fingerprint\ssh\example-model\best_model.pyt")

    ap.add_argument("--train_subjects", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # ✅ Windows + 预加载缓存 强烈建议 0
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--emb_dim", type=int, default=256)

    ap.add_argument("--pretrain_classes", type=int, default=8000)

    ap.add_argument("--loss", type=str, default="arcface", choices=["arcface", "cosface", "triplet"])
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--fuse_mode", type=str, default="concat", choices=["concat", "sum"])

    ap.add_argument("--scale", type=float, default=64.0)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--triplet_margin", type=float, default=0.2)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="./finetuned_sd4_metric")
    ap.add_argument("--impostor_pairs", type=int, default=50000)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available -> CPU")
        device = "cpu"

    subject_ids = list(range(args.train_subjects))

    # ✅ 训练先关增强：你现在要测速/确认GPU
    train_ds = NISTSD4MetricDataset(args.data_dir, subject_ids, is_train=True, augment=True)

    # ✅ val 复用 train 的 preload cache（避免二次 preload）
    val_ds = NISTSD4MetricDataset(
        args.data_dir,
        subject_ids,
        is_train=False,
        augment=False,
        shared_cache=train_ds.img_by_key
    )
    logger.info(f"[Preload] val reuse shared cache: {len(val_ds.img_by_key)}")

    K = 2
    P = max(2, args.batch_size // K)
    sampler = BalancedBatchSampler(train_ds.labels, n_classes=P, n_samples=K)

    train_loader = TorchDataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=False,
    )

    val_loader = TorchDataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False
    )

    model = load_backbone(args.pretrained_path, device=device, emb_dim=args.emb_dim, pretrain_classes=args.pretrain_classes)
    model.train()

    num_classes = train_ds.num_classes
    fused_dim = args.emb_dim * 2 if args.fuse_mode == "concat" else args.emb_dim

    ce = nn.CrossEntropyLoss()
    if args.loss == "arcface":
        head = ArcMarginProduct(fused_dim, num_classes, s=float(args.scale), m=float(args.margin)).to(device)
        triplet = None
    elif args.loss == "cosface":
        head = CosFace(fused_dim, num_classes, s=float(args.scale), m=float(args.margin)).to(device)
        triplet = None
    else:
        head = None
        triplet = BatchHardTripletLoss(margin=float(args.triplet_margin)).to(device)

    params = list(model.parameters())
    if head is not None:
        params += list(head.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_eer = 1.0
    best_path = os.path.join(args.outdir, f"best_{args.loss}.pth")
    last_path = os.path.join(args.outdir, f"last_{args.loss}.pth")

    logger.info(
        f"[Config] loss={args.loss} classes={num_classes} P={P} K={K} "
        f"fuse={args.fuse_mode} alpha={args.alpha} "
        f"num_workers={args.num_workers} impostor_pairs={args.impostor_pairs}"
    )

    # ✅ GPU 检查（你要的）
    logger.info(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    logger.info(f"model device = {next(model.parameters()).device}")
    if head is not None:
        logger.info(f"head device = {next(head.parameters()).device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        if head is not None:
            head.train()

        losses = []
        r1_logs = []  # ✅ 每 20 step 记一次，用来算 avg_R@1（否则你原来永远是0）

        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.long)

            # ✅ 训练第一个 batch 检查输入设备（你要的）
            if step == 0 and epoch == 1:
                logger.info(f"[DeviceCheck] xb device = {xb.device}, yb device = {yb.device}")
                if head is not None:
                    logger.info(f"[DeviceCheck] head device = {next(head.parameters()).device}")

            optimizer.zero_grad(set_to_none=True)

            out = model(xb)
            tex, minu = extract_tex_minu_embeddings(out)
            emb = fuse_embedding(tex, minu, alpha=float(args.alpha), mode=args.fuse_mode)

            if args.loss in ("arcface", "cosface"):
                logits = head(emb, yb)
                loss = ce(logits, yb)
            else:
                loss = triplet(emb, yb)

            loss.backward()
            optimizer.step()

            loss_val = float(loss.detach().cpu())
            losses.append(loss_val)

            # ✅ 只在每 20 step 算一次 R@1（大幅提速）
            if step % 20 == 0:
                r1 = batch_recall_at_1(emb.detach(), yb.detach())
                r1_logs.append(r1)
                logger.info(
                    f"Epoch {epoch:03d} Step {step:04d}/{len(train_loader)} "
                    f"loss={loss_val:.4f} batch_R@1={r1:.3f}"
                )

        eer = eval_eer(
            model,
            val_loader,
            device=device,
            alpha=float(args.alpha),
            fuse_mode=args.fuse_mode,
            impostor_pairs=int(args.impostor_pairs),
        )

        tr_loss = float(np.mean(losses)) if losses else 0.0
        tr_r1 = float(np.mean(r1_logs)) if r1_logs else 0.0
        logger.info(f"[Epoch {epoch:03d}/{args.epochs}] avg_loss={tr_loss:.4f} avg_R@1={tr_r1:.3f} val_EER={eer:.4f}")

        if eer < best_eer:
            best_eer = eer
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "head_state_dict": head.state_dict() if head is not None else None,
                    "args": vars(args),
                    "best_eer": best_eer,
                },
                best_path
            )
            logger.info(f"[Saved] best -> {best_path} (EER={best_eer:.4f})")

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "head_state_dict": head.state_dict() if head is not None else None,
            "args": vars(args),
            "best_eer": best_eer,
        },
        last_path
    )
    logger.info(f"[Saved] last -> {last_path}")


if __name__ == "__main__":
    main()
