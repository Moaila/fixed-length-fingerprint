import os
import sys
import argparse
import numpy as np
import cv2
import torch
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size

#python run_code.py --model ./models/best_model.pyt --img ./data/fingerprints/test/1_8.tif --outdir ./output/2 --minu-threshold 0.01 --nms-size 7

def ensure_u8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    return np.clip(x, 0, 255).astype(np.uint8)

def make_loader(
    crop_size,
    angle,
    roi_mode="NONE",
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
    轻量化预处理逻辑：
    - 读灰度图
    - CLAHE增强
    - 直接pad+resize到DeepPrint输入尺寸
    返回: 可调用的加载函数
    """
    def _load(filepath: str):
        # 1. 读取灰度图
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {filepath}")
        img = ensure_u8(img)

        # 2. CLAHE增强
        if clahe:
            img = cv2.createCLAHE(
                clipLimit=float(clahe_clip),
                tileGridSize=tuple(clahe_grid),
            ).apply(img)

        # 保存原始预处理后的图像（用于细节点掩码）
        img_original = img.copy()

        # 3. 直接pad+resize到目标尺寸
        x = pad_and_resize_to_deepprint_input_size(img, fill=float(fill))

        # 4. 生成可视化用的 uint8
        if torch.is_tensor(x):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)

        # 处理维度
        if x_np.ndim == 3 and x_np.shape[0] == 1:
            x_np2 = x_np[0]
        elif x_np.ndim == 2:
            x_np2 = x_np
        else:
            x_np2 = x_np.squeeze()

        # 转换为uint8
        if x_np2.max() <= 1.5:
            x_u8 = (x_np2 * 255.0).clip(0, 255).astype(np.uint8)
        else:
            x_u8 = x_np2.clip(0, 255).astype(np.uint8)

        return x, x_u8, img_original  # 新增返回原始预处理图像

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

def disable_dropout(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()

def extract_minutiae_points(minu_map, img_gray, threshold=0.45, nms_size=5):
    """
    提取细节点位置（仅位置，无方向）
    :param minu_map: 细节点图 [6, H, W]
    :param img_gray: 原始灰度图，用于生成掩码
    :param threshold: 响应值阈值
    :param nms_size: 非极大值抑制窗口大小
    :return: 细节点坐标列表 [(x1,y1), (x2,y2), ...]
    """
    # 1. 生成指纹区域掩码
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 调整掩码尺寸匹配细节点图
    mask = cv2.resize(binary, (minu_map.shape[2], minu_map.shape[1])) > 0

    # 2. 计算通道维度最大值（响应值）
    max_val_map = np.max(minu_map, axis=0)  # [H, W]

    # 3. 非极大值抑制
    local_max = ndimage.maximum_filter(max_val_map, size=nms_size) == max_val_map

    # 4. 组合过滤条件
    final_mask = local_max & (max_val_map > threshold) & mask

    # 5. 获取细节点坐标
    coords = np.argwhere(final_mask)  # [N, 2] 格式: (y, x)
    # 转换为 (x, y) 格式，并调整坐标到原始图像尺寸
    h_img, w_img = img_gray.shape
    h_f, w_f = minu_map.shape[1:]
    minutiae_pts = []
    for y, x in coords:
        # 坐标映射到原始图像尺寸
        x_ori = x * (w_img / w_f)
        y_ori = y * (h_img / h_f)
        minutiae_pts.append((int(x_ori), int(y_ori)))

    return minutiae_pts

def save_minutiae_visualization(img_u8, minutiae_pts, save_path):
    """
    保存细节点可视化图（仅位置）
    :param img_u8: 预处理后的指纹图像 uint8
    :param minutiae_pts: 细节点坐标列表
    :param save_path: 保存路径
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(img_u8, cmap='gray')
    
    # 绘制细节点（仅位置，无方向线）
    if minutiae_pts:
        xs, ys = zip(*minutiae_pts)
        plt.scatter(xs, ys, s=35, edgecolors='red', facecolors='none', linewidths=1.2)

    plt.title(f"Detected Minutiae (Total: {len(minutiae_pts)})", fontsize=14)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--outdir", default="./examples/out/minumap_vis")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    
    # 细节点检测参数
    ap.add_argument("--minu-threshold", type=float, default=0.45, help="Minutiae response threshold")
    ap.add_argument("--nms-size", type=int, default=7, help="NMS window size for minutiae")

    # 保留兼容参数
    ap.add_argument("--crop-size", type=int, default=400)
    ap.add_argument("--angle", type=float, default=0.0)
    ap.add_argument("--roi-mode", default="NONE")
    ap.add_argument("--mask-apply-mode", default="white")
    ap.add_argument("--dpi", type=int, default=500)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 设备兼容
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # 1. 加载模型
    model = load_model(args.model, device=device)
    model.train()
    disable_dropout(model)

    # 2. 创建加载器并处理图片
    loader = make_loader(
        crop_size=args.crop_size,
        angle=args.angle,
        roi_mode=args.roi_mode,
        mask_apply_mode=args.mask_apply_mode,
        dpi=args.dpi,
    )
    x, x_u8, img_original = loader(args.img)  # 获取原始预处理图像

    # 3. 构造模型输入张量
    if torch.is_tensor(x):
        xt = x
    else:
        xt = torch.from_numpy(np.asarray(x))

    if xt.ndim == 2:
        xt = xt.unsqueeze(0)  # [1,H,W]
    if xt.ndim == 3:
        xt = xt.unsqueeze(0)  # [B=1,C=1,H,W]

    # 补丁：避免batch维度被挤掉
    if xt.shape[0] == 1:
        xt = xt.repeat(2, 1, 1, 1)  # [2,1,H,W]

    xt = xt.to(device=device, dtype=torch.float32)

    # 4. 前向传播
    with torch.no_grad():
        out = model(xt)

    if not hasattr(out, "minutia_maps") or out.minutia_maps is None:
        raise RuntimeError(
            "Model output has no minutia_maps. "
            "This usually means you are not using TexMinu model OR forward did not enter training-output branch."
        )

    minu_tensor = out.minutia_maps  # [B,6,128,128]
    minu = minu_tensor[0].detach().float().cpu().numpy()  # [6,128,128]

    # 5. 保存各类输出（原有功能保留）
    np.save(os.path.join(args.outdir, "minu_map_raw.npy"), minu)

    # 6. 单通道保存
    for k in range(6):
        ch = minu[k]
        chn = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)
        cv2.imwrite(os.path.join(args.outdir, f"minu_ch{k}.png"), (chn * 255).astype(np.uint8))

    # 7. 通道最大值融合图
    merged = np.max(minu, axis=0)
    merged = (merged - merged.min()) / (merged.max() - merged.min() + 1e-6)
    merged_u8 = (merged * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.outdir, "minu_merged.png"), merged_u8)

    # 8. 叠加原图可视化（原有）
    heat_448 = cv2.resize(merged_u8, (x_u8.shape[1], x_u8.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(heat_448, cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(x_u8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.65, heat_color, 0.35, 0.0)
    cv2.imwrite(os.path.join(args.outdir, "input_448.png"), x_u8)
    cv2.imwrite(os.path.join(args.outdir, "overlay.png"), overlay)

    # 9. 新增：提取并可视化细节点（仅位置）
    print(f"\nExtracting minutiae points with threshold={args.minu_threshold}, nms_size={args.nms_size}")
    minutiae_pts = extract_minutiae_points(
        minu_map=minu,
        img_gray=img_original,
        threshold=args.minu_threshold,
        nms_size=args.nms_size
    )
    
    # 保存细节点可视化图
    minutiae_vis_path = os.path.join(args.outdir, "minutiae_points.png")
    save_minutiae_visualization(x_u8, minutiae_pts, minutiae_vis_path)
    
    # 保存细节点坐标到txt文件
    coords_path = os.path.join(args.outdir, "minutiae_coords.txt")
    with open(coords_path, "w") as f:
        f.write("x,y\n")
        for x_pt, y_pt in minutiae_pts:
            f.write(f"{x_pt},{y_pt}\n")
    
    print(f"✅ Minutiae detection completed!")
    print(f"   - Total points: {len(minutiae_pts)}")
    print(f"   - Visualization saved to: {minutiae_vis_path}")
    print(f"   - Coordinates saved to: {coords_path}")
    print(f"\nAll outputs saved to: {args.outdir}")

if __name__ == "__main__":
    main()