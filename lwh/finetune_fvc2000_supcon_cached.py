# -*- coding: utf-8 -*-
"""
finetune_fvc2000_supcon_cached_v3.py

Route B (cached): build_fvc2000_input_cache_gmfs.py 先把输入(299x299或DeepPrint输入尺寸)缓存到磁盘，
这里直接读 cache 做 SupCon 微调，避免 DataLoader 在训练时做 GMFS/裁剪/CLAHE 导致 CPU 卡死。

特点：
- 兼容 meta.jsonl 不同字段命名（自动找 input 文件路径 & identity(db,finger)）
- 相对路径自动以 meta.jsonl 所在目录为基准补齐
- 统计/打印跳过原因（你现在 cached items=0 的根因立刻暴露）
- GPU 训练，pin_memory + persistent_workers
"""

import os
import json
import math
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

# Make flx importable (repo root)
import sys
sys.path.append(os.getcwd())

from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu


# --------------------------
# utils
# --------------------------
def seed_all(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_csv_floats(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_l2(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


def resolve_path(p: str, meta_base: Path) -> Path:
    p = str(p)
    pp = Path(p)
    if pp.is_absolute():
        return pp
    # 相对路径：以 meta.jsonl 所在目录为基准
    return (meta_base / pp).resolve()


# --------------------------
# meta reader (robust)
# --------------------------
CAND_PATH_KEYS = [
    "cache_png", "cache_path", "input_path", "input", "path", "file", "npy", "input_npy", "cached"
]
CAND_DB_KEYS = ["db", "db_name", "dbname", "database"]
CAND_FINGER_KEYS = ["finger", "finger_id", "fid", "subject", "person", "class", "identity_finger"]
CAND_IMP_KEYS = ["imp", "impression", "idx", "sample", "image_id"]


def pick_first(d: Dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None and str(d[k]).strip() != "":
            return k
    return None


def load_cached_items(meta_jsonl: Path, verbose_skip: bool = True, max_print_skip: int = 30):
    meta_base = meta_jsonl.parent
    items = []
    skip_stats = {}
    printed = 0

    with meta_jsonl.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                skip_stats["bad_json"] = skip_stats.get("bad_json", 0) + 1
                if verbose_skip and printed < max_print_skip:
                    print(f"[Skip] line {line_idx}: bad_json")
                    printed += 1
                continue

            if isinstance(obj, dict) and obj.get("error", None):
                skip_stats["meta_has_error"] = skip_stats.get("meta_has_error", 0) + 1
                if verbose_skip and printed < max_print_skip:
                    print(f"[Skip] line {line_idx}: meta_has_error -> {obj.get('error')}")
                    printed += 1
                continue

            kpath = pick_first(obj, CAND_PATH_KEYS)
            if kpath is None:
                skip_stats["no_path_key"] = skip_stats.get("no_path_key", 0) + 1
                if verbose_skip and printed < max_print_skip:
                    print(f"[Skip] line {line_idx}: no_path_key, keys={list(obj.keys())[:12]}")
                    printed += 1
                continue

            p = resolve_path(obj[kpath], meta_base)
            if not p.exists():
                skip_stats["path_not_exist"] = skip_stats.get("path_not_exist", 0) + 1
                if verbose_skip and printed < max_print_skip:
                    print(f"[Skip] line {line_idx}: path_not_exist -> {p}")
                    printed += 1
                continue

            kdb = pick_first(obj, CAND_DB_KEYS)
            kfg = pick_first(obj, CAND_FINGER_KEYS)

            # 允许 meta 里直接给 identity
            identity = obj.get("identity", None)
            if identity is None:
                if kdb is None or kfg is None:
                    skip_stats["no_identity_keys"] = skip_stats.get("no_identity_keys", 0) + 1
                    if verbose_skip and printed < max_print_skip:
                        print(f"[Skip] line {line_idx}: no_identity_keys, keys={list(obj.keys())[:12]}")
                        printed += 1
                    continue
                db = str(obj[kdb])
                finger = int(obj[kfg])
                identity = f"{db}__{finger:03d}"
            else:
                identity = str(identity)

            items.append((p, identity))

    if verbose_skip:
        print("[Meta] skip_stats =", skip_stats)
        if len(items) > 0:
            print("[Meta] example item:", items[0][0], items[0][1])

    return items


# --------------------------
# cache dataset
# --------------------------
def load_cached_input(path: Path) -> np.ndarray:
    """
    支持：
    - .npy : (H,W) or (1,H,W) float/uint8
    - .png/.jpg : uint8
    最终返回 float32 [1,H,W] in [0,1]
    """
    suf = path.suffix.lower()
    if suf == ".npy":
        x = np.load(str(path), allow_pickle=False)
    else:
        import cv2
        x = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if x is None:
            raise ValueError(f"Failed to read image: {path}")

    x = np.asarray(x)
    if x.ndim == 2:
        x = x[None, ...]
    elif x.ndim == 3:
        # CHW or HWC?
        if x.shape[0] == 1:
            pass
        elif x.shape[2] == 1:
            x = np.transpose(x, (2, 0, 1))
        else:
            # 取灰度第一通道
            x = x[0:1, ...]
    else:
        raise ValueError(f"Bad cached shape: {x.shape} @ {path}")

    x = x.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    x = np.clip(x, 0.0, 1.0)
    return x


def aug_pair(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    轻量 augment：尽量别让 CPU 搞 cv2 大旋转（否则又回到 CPU 瓶颈）
    这里全部 torch 上做（在 DataLoader 出来后再 .to(cuda) 前也能跑，但很快）
    """
    def _one(z: torch.Tensor):
        # z: [1,H,W]
        # brightness/contrast jitter
        if torch.rand(()) < 0.9:
            a = 0.9 + 0.2 * torch.rand(())
            b = (-0.05 + 0.10 * torch.rand(()))
            z = z * a + b
        # gaussian noise
        if torch.rand(()) < 0.9:
            z = z + 0.02 * torch.randn_like(z)
        z = z.clamp(0, 1)
        # random erasing (small)
        if torch.rand(()) < 0.3:
            _, H, W = z.shape
            rh = int(H * (0.05 + 0.10 * torch.rand(())))
            rw = int(W * (0.05 + 0.10 * torch.rand(())))
            y0 = int((H - rh) * torch.rand(()))
            x0 = int((W - rw) * torch.rand(()))
            z[:, y0:y0+rh, x0:x0+rw] = torch.rand(())
        return z

    return _one(x.clone()), _one(x.clone())


class CachedSupConDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, str]], id2idx: Dict[str, int]):
        self.items = items
        self.id2idx = id2idx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, identity = self.items[idx]
        x = load_cached_input(p)  # [1,H,W] float32 [0,1]
        y = self.id2idx[identity]
        x = torch.from_numpy(x)   # float32
        x1, x2 = aug_pair(x)
        return x1, x2, y


# --------------------------
# PK sampler (batch = P*K)
# --------------------------
class PKBatchSampler(torch.utils.data.Sampler[List[int]]):
    def __init__(self, labels: List[int], P: int, K: int, drop_last: bool = True):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.P = int(P)
        self.K = int(K)
        self.drop_last = drop_last

        self.cls_to_indices: Dict[int, np.ndarray] = {}
        for i, y in enumerate(self.labels):
            self.cls_to_indices.setdefault(int(y), []).append(i)
        for c in list(self.cls_to_indices.keys()):
            self.cls_to_indices[c] = np.asarray(self.cls_to_indices[c], dtype=np.int64)

        self.classes = np.asarray(list(self.cls_to_indices.keys()), dtype=np.int64)
        self.num_classes = len(self.classes)

    def __iter__(self):
        rng = np.random.default_rng()
        # 无穷生成，由 DataLoader/epoch 控制 steps
        while True:
            # 选 P 个类
            if self.num_classes >= self.P:
                chosen = rng.choice(self.classes, size=self.P, replace=False)
            else:
                chosen = rng.choice(self.classes, size=self.P, replace=True)

            batch = []
            for c in chosen:
                idxs = self.cls_to_indices[int(c)]
                if len(idxs) >= self.K:
                    pick = rng.choice(idxs, size=self.K, replace=False)
                else:
                    pick = rng.choice(idxs, size=self.K, replace=True)
                batch.extend(pick.tolist())
            yield batch

    def __len__(self):
        # 不严格：给一个很大的值
        return 10**9


# --------------------------
# SupCon loss (stable)
# --------------------------
def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    features: [B, 2, D]  (two views)
    labels:   [B]
    """
    device = features.device
    B = features.shape[0]
    V = features.shape[1]  # 2
    D = features.shape[2]

    # [2B, D]
    feats = features.reshape(B * V, D)
    feats = safe_l2(feats, dim=1)

    # labels expand [2B]
    labs = labels.reshape(B, 1).repeat(1, V).reshape(B * V)

    # similarity [2B,2B]
    sim = torch.matmul(feats, feats.T) / float(temperature)
    sim = sim - sim.max(dim=1, keepdim=True).values  # stability

    # mask positives (exclude self)
    labs_eq = labs.unsqueeze(0) == labs.unsqueeze(1)   # [2B,2B]
    self_mask = torch.eye(B * V, dtype=torch.bool, device=device)
    pos_mask = labs_eq & (~self_mask)

    # log_prob
    exp_sim = torch.exp(sim) * (~self_mask)  # remove self from denom
    denom = exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_prob = sim - torch.log(denom)

    # mean of positives per anchor
    pos_count = pos_mask.sum(dim=1).clamp_min(1)  # avoid div0
    loss = -(log_prob * pos_mask).sum(dim=1) / pos_count
    return loss.mean()


# --------------------------
# model loading + freeze policy
# --------------------------
@torch.no_grad()
def load_deepprint_model(model_path: str, device: str, n_classes: int = 8000, emb_dim: int = 256):
    extractor = get_DeepPrint_TexMinu(n_classes, emb_dim)
    ckpt = torch.load(model_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    md = extractor.model.state_dict()
    matched = 0
    for k, v in sd.items():
        if k in md and hasattr(v, "shape") and v.shape == md[k].shape:
            md[k] = v
            matched += 1
    extractor.model.load_state_dict(md)
    extractor.model.to(device).eval()
    return extractor.model, matched, len(md)


def set_trainable(model: nn.Module, mode: str = "last"):
    """
    last: 只训最后若干层（更稳，不至于把预训练搞崩）
    all : 全部训练
    """
    mode = mode.lower().strip()
    for p in model.parameters():
        p.requires_grad = False

    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    # "last": 尽量只放开 embedding heads / 最后几层
    # 由于 flx 架构你本地版本可能略不同，这里用“名字匹配”策略更鲁棒：
    allow_keywords = [
        "fc", "classifier", "head", "embedding", "emb", "projection", "proj",
        "tex", "minu", "fusion"
    ]
    for name, p in model.named_parameters():
        if any(k in name.lower() for k in allow_keywords):
            p.requires_grad = True


def count_trainable(model: nn.Module):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    a = sum(p.numel() for p in model.parameters())
    return t, a


# --------------------------
# forward to get tex/minu embeddings
# --------------------------
@torch.no_grad()
def forward_embeddings(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    这里直接调用 model(x)，按你 flx 的 DeepPrint_TexMinu forward 输出结构来取：
    - 有的版本返回 (tex_emb, minu_emb) 或 dict
    - 这里做兼容
    """
    out = model(x)

    # 常见情况1：tuple/list
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        tex, minu = out[0], out[1]
        return tex, minu

    # 常见情况2：dict
    if isinstance(out, dict):
        # 你本地 key 名可能不同，尽量覆盖
        for tk in ["texture_emb", "tex_emb", "texture", "tex"]:
            if tk in out:
                tex = out[tk]
                break
        else:
            raise RuntimeError(f"Cannot find texture emb in dict keys={list(out.keys())}")

        for mk in ["minutiae_emb", "minu_emb", "minutiae", "minu"]:
            if mk in out:
                minu = out[mk]
                break
        else:
            raise RuntimeError(f"Cannot find minutiae emb in dict keys={list(out.keys())}")

        return tex, minu

    raise RuntimeError(f"Unknown model output type: {type(out)}")


def forward_fused(model: nn.Module, x: torch.Tensor, alpha: float = 5.0) -> torch.Tensor:
    tex, minu = forward_embeddings(model, x)
    tex = safe_l2(tex, dim=1)
    minu = safe_l2(minu, dim=1)
    fused = tex + float(alpha) * minu
    fused = safe_l2(fused, dim=1)
    return fused


# --------------------------
# main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-meta", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--trainable", default="last", choices=["last", "all"])

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps-per-epoch", type=int, default=300, help="每个 epoch 跑多少个 batch（避免无限 sampler）")

    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--temperature", type=float, default=0.07)

    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--val-imp", type=int, default=8, help="每个 identity 留 val_imp 张做 val（其余 train）")
    ap.add_argument("--print-skip", action="store_true", default=False)

    args = ap.parse_args()
    seed_all(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA not available, fallback to CPU.")
        device = "cpu"
    use_amp = (device == "cuda") and (not args.no_amp)

    if device == "cuda":
        print(f"[Device] cuda | amp={use_amp}")
        print("[CUDA]", torch.cuda.get_device_name(0))

    meta_jsonl = Path(args.cache_meta).resolve()
    items = load_cached_items(meta_jsonl, verbose_skip=args.print_skip)

    print(f"[Data] cached items = {len(items)}")
    if len(items) == 0:
        print("\n[!!!] 你现在 cached items=0 的原因只可能是：")
        print("  1) meta.jsonl 里路径字段名不是 cache_png/input/...（脚本已打印 skip_stats）")
        print("  2) meta 里写的是相对路径，但不是相对 meta.jsonl 的目录")
        print("  3) meta 里写的路径在当前机器不存在")
        print("\n你先执行：")
        print(f"  head -n 3 {meta_jsonl}")
        print("把前三行贴出来我就能一眼定位字段名。")
        return

    # build identity -> indices
    identities = [it[1] for it in items]
    uniq_ids = sorted(list(set(identities)))
    id2idx = {u: i for i, u in enumerate(uniq_ids)}
    print(f"[Data] identities(db,finger) = {len(uniq_ids)}")

    # split per identity: last val_imp samples -> val
    # 先按 identity 聚合
    by_id: Dict[str, List[int]] = {}
    for i, (_, ident) in enumerate(items):
        by_id.setdefault(ident, []).append(i)

    train_idx, val_idx = [], []
    for ident, idxs in by_id.items():
        idxs = sorted(idxs)
        if len(idxs) <= args.val_imp:
            # 太少就全进 train
            train_idx.extend(idxs)
        else:
            val_idx.extend(idxs[-args.val_imp:])
            train_idx.extend(idxs[:-args.val_imp])

    print(f"[Split] train={len(train_idx)} val={len(val_idx)} (val_imp={args.val_imp})")

    # prepare labels list for sampler
    train_labels = [id2idx[items[i][1]] for i in train_idx]

    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]

    train_ds = CachedSupConDataset(train_items, id2idx)
    val_ds = CachedSupConDataset(val_items, id2idx)

    # infinite PK sampler
    assert args.batch_size == args.P * args.K, "batch-size 必须等于 P*K"
    pk_sampler = PKBatchSampler(train_labels, P=args.P, K=args.K)

    def collate_fn(batch):
        x1 = torch.stack([b[0] for b in batch], dim=0)  # [B,1,H,W]
        x2 = torch.stack([b[1] for b in batch], dim=0)
        y = torch.tensor([b[2] for b in batch], dtype=torch.long)
        return x1, x2, y

    train_loader = DataLoader(
        train_ds,
        batch_sampler=pk_sampler,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn,
    )

    # load model
    model, matched, total = load_deepprint_model(args.model, device=device)
    print(f"[Model] loaded params matched: {matched}/{total}")

    set_trainable(model, args.trainable)
    ntrain, nall = count_trainable(model)
    print(f"[Model] trainable={ntrain/1e6:.2f}M / total={nall/1e6:.2f}M ({args.trainable})")

    # optimizer
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    best_path = Path(args.outdir) / "best_model_supcon.pyt"
    last_path = Path(args.outdir) / "last_model_supcon.pyt"

    # sanity batch
    x1b, x2b, yb = next(iter(train_loader))
    print("[Sanity] one batch:", tuple(x1b.shape), tuple(x2b.shape), tuple(yb.shape))
    print("[Sanity] unique identities in batch =", int(len(torch.unique(yb))), f"(expect ~P={args.P})")

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []

        it = iter(train_loader)
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {ep:03d}/{args.epochs}", ncols=100)
        for _ in pbar:
            x1, x2, y = next(it)
            x1 = x1.to(device=device, dtype=torch.float32, non_blocking=True)
            x2 = x2.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    z1 = forward_fused(model, x1, alpha=args.alpha)
                    z2 = forward_fused(model, x2, alpha=args.alpha)
                    feats = torch.stack([z1, z2], dim=1)  # [B,2,D]
                    loss = supcon_loss(feats, y, temperature=args.temperature)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                z1 = forward_fused(model, x1, alpha=args.alpha)
                z2 = forward_fused(model, x2, alpha=args.alpha)
                feats = torch.stack([z1, z2], dim=1)
                loss = supcon_loss(feats, y, temperature=args.temperature)
                loss.backward()
                opt.step()

            lv = float(loss.detach().cpu().item())
            losses.append(lv)
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        train_loss = float(np.mean(losses)) if len(losses) else float("inf")

        # val
        model.eval()
        vlosses = []
        with torch.no_grad():
            for x1, x2, y in tqdm(val_loader, desc=f"Val {ep:03d}", ncols=100):
                x1 = x1.to(device=device, dtype=torch.float32, non_blocking=True)
                x2 = x2.to(device=device, dtype=torch.float32, non_blocking=True)
                y = y.to(device=device, non_blocking=True)

                z1 = forward_fused(model, x1, alpha=args.alpha)
                z2 = forward_fused(model, x2, alpha=args.alpha)
                feats = torch.stack([z1, z2], dim=1)
                vl = supcon_loss(feats, y, temperature=args.temperature)
                vlosses.append(float(vl.detach().cpu().item()))

        val_loss = float(np.mean(vlosses)) if len(vlosses) else float("inf")
        dt = time.time() - t0
        print(f"[Epoch {ep:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={dt:.1f}s")

        # save last
        torch.save({"model_state_dict": model.state_dict()}, str(last_path))

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict()}, str(best_path))
            print(f"[Saved] best -> {best_path} (val_loss={best_val:.4f})")

    print("[Done] best =", best_path)
    print("[Done] last =", last_path)


if __name__ == "__main__":
    main()
