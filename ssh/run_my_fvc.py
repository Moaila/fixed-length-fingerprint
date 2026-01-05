import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from sklearn.metrics import roc_curve


sys.path.append(os.getcwd())
from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu
from flx.setup.datasets import get_fvc2004_db1a
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
from flx.data.image_loader import FVC2004Loader

# ========================= 路径配置 =========================
MODEL_PATH = "./models/best_model.pyt"
DATA_DIR   = "./data/fingerprints/FVC2000/Dbs/Db1_a"

# ========================= 视角增强（TTA） =========================
TTA_SCALES = [340, 370, 400, 430]
TTA_ANGLES = [-15, 0, 15]
V = len(TTA_SCALES) * len(TTA_ANGLES)

# ========================= 融合参数 =========================
TARGET_FMR = 2e-5
ALPHA       = 5.0

TEX_PAIR_TOPK  = 6

TEX_SPIKE_CLIP_ENABLE = True
TEX_DELTA = 0.08

# ============================================================
#                     评估工具
# ============================================================

def compute_eer_from_scores(y_scores, y_true):
    fpr, tpr, thr = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thr[idx])

def eval_at_target_from_scores(y_scores, y_true, target_fmr):
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
#                   图像预处理
# ============================================================

def make_loader(crop_size_target, angle_target):
    @staticmethod
    def _load(filepath):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {filepath}")
            
        img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)

        if angle_target != 0:
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_target, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=255)

        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.argwhere(th > 0)

        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        if len(coords) > 0:
            cy, cx = coords.mean(0).astype(int)

        cs = crop_size_target
        sy = max(0, min(cy - cs // 2, img.shape[0] - cs))
        sx = max(0, min(cx - cs // 2, img.shape[1] - cs))
        img = img[sy:sy + cs, sx:sx + cs]

        return pad_and_resize_to_deepprint_input_size(img, fill=1.0)

    return _load

# ============================================================
#                         模型与特征
# ============================================================

@torch.no_grad()
def load_model():
    extractor = get_DeepPrint_TexMinu(8000, 256)
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    md = extractor.model.state_dict()
    md.update({k: v for k, v in sd.items() if k in md and v.shape == md[k].shape})
    extractor.model.load_state_dict(md)
    extractor.model.eval().cpu()
    return extractor

@torch.no_grad()
def extract_views_tex_minu(extractor):
    tex_views, minu_views = [], []
    for s in TTA_SCALES:
        for a in TTA_ANGLES:
            print(f"[View] scale={s} angle={a}")
            FVC2004Loader._load_image = make_loader(s, a)
            ds = get_fvc2004_db1a(DATA_DIR)
            
            if len(ds) == 0:
                raise ValueError("Dataset is empty. Check DATA_DIR.")

            tex, minu = extractor.extract(ds)

            t = tex._array if torch.is_tensor(tex._array) else torch.from_numpy(tex._array)
            m = minu._array if torch.is_tensor(minu._array) else torch.from_numpy(minu._array)

            tex_views.append(F.normalize(t.cpu(), p=2, dim=1))   # [800,256]
            minu_views.append(F.normalize(m.cpu(), p=2, dim=1))  # [800,256]

    return torch.stack(tex_views, dim=1), torch.stack(minu_views, dim=1)  # [800,V,256]

# ============================================================
#          multi-view view-pair robust score
# ============================================================

@torch.no_grad()
def viewpair_topkmean(enroll_views, probe_views, topk=6):
    sim = torch.einsum("nvd,mwd->nmvw", enroll_views, probe_views)  # [N,M,V,V]
    flat = sim.reshape(sim.shape[0], sim.shape[1], -1)             # [N,M,V*V]
    kk = min(topk, flat.shape[-1])
    return flat.topk(kk, dim=-1).values.mean(dim=-1)               # [N,M]

@torch.no_grad()
def viewpair_gapaware(enroll_views, probe_views, lam=0.35):
    sim = torch.einsum("nvd,mwd->nmvw", enroll_views, probe_views)  # [N,M,V,V]
    flat = sim.reshape(sim.shape[0], sim.shape[1], -1)             # [N,M,V*V]
    top2 = flat.topk(2, dim=-1).values                              # [N,M,2]
    top1 = top2[..., 0]
    top2v = top2[..., 1]
    gap = top1 - top2v
    return top1 - lam * gap                                         # [N,M]

@torch.no_grad()
def viewpair_top2_clip(enroll_views, probe_views, delta=0.08):
    sim = torch.einsum("nvd,mwd->nmvw", enroll_views, probe_views)
    flat = sim.reshape(sim.shape[0], sim.shape[1], -1)
    top2 = flat.topk(2, dim=-1).values
    m1 = top2[..., 0]
    m2 = top2[..., 1]
    return torch.minimum(m1, m2 + float(delta))                    # [N,M]

# ============================================================
#                       主流程
# ============================================================

def main():
    extractor = load_model()

    # 1. 提取所有 800 张图的特征 (GPU)
    # 结构: [800, V, 256] -> 800个样本(100指x8次), V个视角, 256维
    tex_views, minu_views = extract_views_tex_minu(extractor)   
    device = tex_views.device

    y_scores_gen = []
    y_scores_imp = []

    print("\n================ 8-FOLD 1:1 EVALUATION ================\n")
    print("Mode: One Enroll vs All Others (Leave-One-Out)")
    
    # 8折交叉验证：每次取每个手指的第 k 次按压作为注册（Template），其余作为查询（Probe）
    for k in range(8):
        # --- 构建索引 ---
        # 注册集: 100个手指，每个手指取第 k 张
        enroll_idx = (torch.arange(100, device=device) * 8 + k)  # [100]
        enroll_labels = torch.arange(100, device=device)         # [100] 对应ID 0..99
        
        # 查询集: 100个手指，每个手指取除了 k 以外的 7 张
        probe_idx_list = []
        probe_label_list = []
        for f in range(100):
            base = f * 8
            for imp in range(8):
                if imp == k: continue 
                probe_idx_list.append(base + imp)
                probe_label_list.append(f)

        probe_idx = torch.tensor(probe_idx_list, device=device, dtype=torch.long)      # [700]
        probe_labels = torch.tensor(probe_label_list, device=device, dtype=torch.long) # [700]

        # --- 提取对应的特征子集 ---
        enroll_tex = tex_views[enroll_idx]      # [100, V, 256]
        probe_tex  = tex_views[probe_idx]       # [700, V, 256]
        enroll_minu = minu_views[enroll_idx]
        probe_minu  = minu_views[probe_idx]

        # --- 计算 N x M 相似度矩阵 ---
        # 这一步计算了 100个注册样本 与 700个查询样本 之间的所有可能两两配对
        if TEX_SPIKE_CLIP_ENABLE:
            s_tex_mat = viewpair_top2_clip(enroll_tex, probe_tex, delta=TEX_DELTA) 
        else:
            s_tex_mat = viewpair_topkmean(enroll_tex, probe_tex, topk=TEX_PAIR_TOPK)

        s_minu_mat = viewpair_gapaware(enroll_minu, probe_minu)
        
        # 融合分数矩阵
        # [100, 700]
        scores_mat = s_tex_mat + float(ALPHA) * s_minu_mat
        
        # --- 利用掩码提取 1:1 配对 ---
        # enroll_labels: [100], probe_labels: [700]
        mask_genuine = (enroll_labels.unsqueeze(1) == probe_labels.unsqueeze(0)) # [100, 700]
        
        # 提取正样本分数 (Same Finger, Different Impression)
        # 数量: 100 * 7 = 700 个
        gen_scores = scores_mat[mask_genuine].detach().cpu().numpy()
        y_scores_gen.append(gen_scores)
        
        # 提取负样本分数 (Different Finger)
        # 数量: 100 * 700 - 700 = 69,300 个
        imp_scores = scores_mat[~mask_genuine].detach().cpu().numpy()
        y_scores_imp.append(imp_scores)

        print(f"[Fold {k}] Gen: {len(gen_scores)} pairs | Imp: {len(imp_scores)} pairs")

    y_scores_gen = np.concatenate(y_scores_gen) # Total: 5,600
    y_scores_imp = np.concatenate(y_scores_imp) # Total: 554,400
    
    y_scores = np.concatenate([y_scores_gen, y_scores_imp])
    y_true = np.concatenate([np.ones_like(y_scores_gen), np.zeros_like(y_scores_imp)])

    print("\n================ 1:1 RESULT ================")
    for t, name in [(0.05, "5%"), (1e-3, "1/1k"), (1e-4, "1/10k"), (2e-5, "1/50k")]:
        fn, th = eval_at_target_from_scores(y_scores, y_true, t)
        print(f"{name:<6}  FFR={fn*100:6.2f}%  thr≈{th:.4f}")

    eer, eer_thr = compute_eer_from_scores(y_scores, y_true)
    fn_50k, th_50k = eval_at_target_from_scores(y_scores, y_true, TARGET_FMR)

    print("-------------------------------------------------------------")
    print(f"[@1/50k] FFR={fn_50k*100:.2f}% thr≈{th_50k:.4f}")
    print(f"EER = {eer*100:.2f}%   (threshold ≈ {eer_thr:.4f})")
    print("=============================================================\n")

if __name__ == "__main__":
    main()