# vis_minumap.py
# DeepPrint TexMinu: 输出 6 通道 minutiae-map，并抽取“点”叠加可视化（带 BN/Dropout 冻结修复）

import os
import sys
import argparse
import csv
import numpy as np
import cv2
import torch

sys.path.append(os.getcwd())

from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size

try:
    import pyfing.simple_api as pf
except Exception:
    pf = None


def ensure_u8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    return np.clip(x, 0, 255).astype(np.uint8)


def get_pyfing_mask(img_u8: np.ndarray, dpi=500, roi_mode="PYFING_GMFS") -> np.ndarray:
    if pf is None:
        raise RuntimeError("pyfing.simple_api import failed, cannot use PYFING_GMFS/PYFING_SUFS.")
    mode = roi_mode.upper()
    if mode == "PYFING_GMFS":
        m = pf.fingerprint_segmentation(img_u8, dpi=dpi, method="GMFS")
    elif mode == "PYFING_SUFS":
        m = pf.fingerprint_segmentation(img_u8, dpi=dpi, method="SUFS")
    else:
        raise ValueError("roi_mode must be PYFING_GMFS or PYFING_SUFS (or NONE).")
    m = np.asarray(m)
    return (m > 0).astype(np.uint8) * 255


def make_loader(
    crop_size,
    angle,
    roi_mode="PYFING_GMFS",
    mask_apply_mode="white",
    dpi=500,
    clahe=True,
    clahe_clip=3.0,
    clahe_grid=(8, 8),
    blur_ksize=5,
    rot_border=255,
    fill=1.0,
):
    """
    复刻你 run_my_fvc4 的关键预处理（简化版）：
    - 灰度读入
    - CLAHE
    - 旋转（可选）
    - ROI mask + mask_apply_mode=white
    - OTSU centroid crop
    - pad + resize 到 DeepPrint 输入 448x448（float 或 tensor）
    返回: (x_for_model, x_u8_448)
    """
    @staticmethod
    def _load(filepath: str):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {filepath}")
        img = ensure_u8(img)

        if clahe:
            img = cv2.createCLAHE(
                clipLimit=float(clahe_clip),
                tileGridSize=tuple(clahe_grid),
            ).apply(img)

        if float(angle) != 0.0:
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), float(angle), 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=int(rot_border))

        if roi_mode.upper() != "NONE":
            mask = get_pyfing_mask(img, dpi=dpi, roi_mode=roi_mode)
            if mask_apply_mode.lower() == "white":
                img = img.copy()
                img[mask == 0] = 255

        # OTSU centroid crop
        k = int(blur_ksize)
        if k <= 0:
            k = 1
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur(img, (k, k), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.argwhere(th > 0)

        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        if len(coords) > 0:
            cy, cx = coords.mean(0).astype(int)

        cs = int(crop_size)
        cs = min(cs, img.shape[0], img.shape[1])
        sy = max(0, min(cy - cs // 2, img.shape[0] - cs))
        sx = max(0, min(cx - cs // 2, img.shape[1] - cs))
        img_crop = img[sy:sy + cs, sx:sx + cs]

        x = pad_and_resize_to_deepprint_input_size(img_crop, fill=float(fill))

        # 生成可视化 448x448 uint8
        if torch.is_tensor(x):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)

        if x_np.ndim == 3 and x_np.shape[0] == 1:
            x_np2 = x_np[0]
        else:
            x_np2 = x_np.squeeze()

        if x_np2.max() <= 1.5:
            x_u8 = (x_np2 * 255.0).clip(0, 255).astype(np.uint8)
        else:
            x_u8 = x_np2.clip(0, 255).astype(np.uint8)

        return x, x_u8

    return _load


def load_model(model_path: str, device: str = "cuda"):
    extractor = get_DeepPrint_TexMinu(8000, 256)
    ckpt = torch.load(model_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    md = extractor.model.state_dict()
    md.update({k: v for k, v in sd.items() if k in md and hasattr(v, "shape") and v.shape == md[k].shape})
    extractor.model.load_state_dict(md)

    extractor.model.to(device)
    return extractor.model


def freeze_dropout_and_bn(model: torch.nn.Module):
    """
    关键修复：
    - model.train() 是为了走 training 分支，拿到 minutia_maps
    - 但 BatchNorm 不能 train（否则会用当前 batch 统计把输出抹平）
    - Dropout 也不能 train（否则输出随机抖动）
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def nms_peaks(prob: np.ndarray, thr: float, topk: int, nms_ksize: int):
    """
    prob: (6,128,128) in [0,1]
    返回 peaks: [(x128,y128,c,score), ...] 按 score 降序
    """
    peaks = []
    k = int(nms_ksize)
    if k <= 0:
        k = 1
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)

    for c in range(prob.shape[0]):
        p = prob[c].astype(np.float32)
        dil = cv2.dilate(p, kernel)
        is_peak = (p >= dil - 1e-8)
        mask = (p >= float(thr)) & is_peak
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        scores = p[ys, xs]
        order = np.argsort(-scores)
        if topk is not None and len(order) > int(topk):
            order = order[: int(topk)]
        for idx in order:
            peaks.append((int(xs[idx]), int(ys[idx]), int(c), float(scores[idx])))

    peaks.sort(key=lambda t: -t[3])
    return peaks


def force_topk_peaks(prob: np.ndarray, topk: int, nms_ksize: int):
    """
    不依赖阈值：每通道强制取 topk 个局部峰（如果峰很少就取到有为止）
    用于你现在这种 prob 很平的时候，至少能看到“模型更偏好哪里”
    """
    peaks = []
    k = int(nms_ksize)
    if k <= 0:
        k = 1
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)

    for c in range(prob.shape[0]):
        p = prob[c].astype(np.float32)
        dil = cv2.dilate(p, kernel)
        is_peak = (p >= dil - 1e-8)
        ys, xs = np.where(is_peak)
        if xs.size == 0:
            continue
        scores = p[ys, xs]
        order = np.argsort(-scores)
        if topk is not None and len(order) > int(topk):
            order = order[: int(topk)]
        for idx in order:
            peaks.append((int(xs[idx]), int(ys[idx]), int(c), float(scores[idx])))

    peaks.sort(key=lambda t: -t[3])
    return peaks


def draw_points_with_orientation(base_gray_448: np.ndarray, peaks, scale=3.5, radius=4, arrow_len=14):
    """
    peaks: [(x128,y128,c,score)]
    """
    bgr = cv2.cvtColor(base_gray_448, cv2.COLOR_GRAY2BGR)

    for (x128, y128, c, s) in peaks:
        x = int(round(x128 * scale))
        y = int(round(y128 * scale))

        # 点（黄圈+红心）
        cv2.circle(bgr, (x, y), radius, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(bgr, (x, y), 1, (0, 0, 255), -1, cv2.LINE_AA)

        # 方向箭头（粗方向 bin，仅用于解释）
        theta_deg = c * (180.0 / 6.0)  # 0,30,60,90,120,150
        theta = np.deg2rad(theta_deg)
        dx = int(round(np.cos(theta) * arrow_len))
        dy = int(round(np.sin(theta) * arrow_len))
        cv2.arrowedLine(bgr, (x, y), (x + dx, y + dy), (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.25)

    return bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--outdir", default="./examples/out/minumap_vis")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    ap.add_argument("--crop-size", type=int, default=400)
    ap.add_argument("--angle", type=float, default=0.0)
    ap.add_argument("--roi-mode", default="PYFING_GMFS")
    ap.add_argument("--mask-apply-mode", default="white")
    ap.add_argument("--dpi", type=int, default=500)

    # 点提取参数
    ap.add_argument("--peak-thr", type=float, default=0.60, help="sigmoid(prob) threshold")
    ap.add_argument("--topk", type=int, default=80, help="max peaks per channel")
    ap.add_argument("--nms-ksize", type=int, default=9, help="NMS dilation kernel size")
    ap.add_argument("--force-topk", type=int, default=30, help="if no peaks by thr, force topK peaks per channel")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # 1) model
    model = load_model(args.model, device=device)

    # 关键：进入 training 分支拿 minutia_maps，但冻结 BN/Dropout
    model.train()
    freeze_dropout_and_bn(model)

    # 2) preprocess
    loader = make_loader(
        crop_size=args.crop_size,
        angle=args.angle,
        roi_mode=args.roi_mode,
        mask_apply_mode=args.mask_apply_mode,
        dpi=args.dpi,
    )
    x, x_u8 = loader(args.img)

    # 3) tensor [B,C,H,W]
    if torch.is_tensor(x):
        xt = x
    else:
        xt = torch.from_numpy(np.asarray(x))

    if xt.ndim == 2:
        xt = xt.unsqueeze(0)  # [1,H,W]
    if xt.ndim == 3:
        xt = xt.unsqueeze(0)  # [B=1,C=1,H,W]

    # 避免 squeeze 把 batch 维挤掉
    if xt.shape[0] == 1:
        xt = xt.repeat(2, 1, 1, 1)

    xt = xt.to(device=device, dtype=torch.float32)

    # 4) forward
    with torch.no_grad():
        out = model(xt)

    if not hasattr(out, "minutia_maps") or out.minutia_maps is None:
        raise RuntimeError("Model output has no minutia_maps (check you are using TexMinu model).")

    logits = out.minutia_maps[0].detach().float().cpu().numpy()  # [6,128,128]
    prob = sigmoid_np(logits)

    # 打印统计，方便你知道阈值该怎么设
    print("prob stats:",
          "min", float(prob.min()),
          "max", float(prob.max()),
          "mean", float(prob.mean()))

    # 5) save raw arrays
    np.save(os.path.join(args.outdir, "minu_map_logits.npy"), logits)
    np.save(os.path.join(args.outdir, "minu_map_prob.npy"), prob)

    # 6) save per-channel prob images
    for k in range(6):
        ch_u8 = (prob[k] * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.outdir, f"minu_prob_ch{k}.png"), ch_u8)

    merged = np.max(prob, axis=0)
    merged_u8 = (merged * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.outdir, "minu_prob_merged.png"), merged_u8)

    # 7) overlay heat
    heat_448 = cv2.resize(merged_u8, (x_u8.shape[1], x_u8.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(heat_448, cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(x_u8, cv2.COLOR_GRAY2BGR)
    overlay_heat = cv2.addWeighted(base_bgr, 0.70, heat_color, 0.30, 0.0)
    cv2.imwrite(os.path.join(args.outdir, "input_448.png"), x_u8)
    cv2.imwrite(os.path.join(args.outdir, "overlay_heat.png"), overlay_heat)

    # 8) peaks (threshold-based)
    peaks = nms_peaks(prob, thr=args.peak_thr, topk=args.topk, nms_ksize=args.nms_ksize)

    # 如果一个点都没有，则启用强制 topk
    if len(peaks) == 0 and args.force_topk and args.force_topk > 0:
        print(f"[WARN] no peaks with thr={args.peak_thr:.3f}. Use force-topk={args.force_topk} per channel.")
        peaks = force_topk_peaks(prob, topk=args.force_topk, nms_ksize=args.nms_ksize)

    # 9) export csv
    csv_path = os.path.join(args.outdir, "points.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x128", "y128", "channel", "score_prob", "x448", "y448"])
        for (x128, y128, c, s) in peaks:
            x448 = int(round(x128 * 3.5))
            y448 = int(round(y128 * 3.5))
            w.writerow([x128, y128, c, f"{s:.6f}", x448, y448])

    # 10) overlay points
    overlay_pts = draw_points_with_orientation(x_u8, peaks, scale=3.5, radius=4, arrow_len=14)
    cv2.imwrite(os.path.join(args.outdir, "overlay_points.png"), overlay_pts)

    print("[OK] saved to:", args.outdir)
    print(" - input_448.png")
    print(" - overlay_heat.png")
    print(" - overlay_points.png")
    print(" - points.csv")
    print("Tip: try --peak-thr 0.7/0.8 or lower 0.55 based on printed prob max.")


if __name__ == "__main__":
    main()
