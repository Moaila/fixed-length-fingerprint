import os
import sys
import argparse
import numpy as np
import cv2
import torch
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

"""
批处理，细节点输出，热力图输出
"""

# 支持的图片格式
SUPPORTED_FORMATS = ('.bmp', '.tif', '.png', '.jpg', '.jpeg', '.tiff')

sys.path.append(os.getcwd())

from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size

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
    """完全复刻单文件版本的预处理逻辑"""
    def _load(filepath: str):
        # 1. 读取灰度图
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {filepath}")
        img = ensure_u8(img)

        # 2. CLAHE增强（和单文件版本完全一致）
        if clahe:
            img = cv2.createCLAHE(
                clipLimit=float(clahe_clip),
                tileGridSize=tuple(clahe_grid),
            ).apply(img)

        # 保存原始预处理后的图像（用于细节点掩码）
        img_original = img.copy()

        # 3. 直接pad+resize到DeepPrint输入尺寸（和单文件版本一致）
        x = pad_and_resize_to_deepprint_input_size(img, fill=float(fill))

        # 4. 生成可视化用的 uint8（完全复刻单文件逻辑）
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

        return x, x_u8, img_original

    return _load

def load_model(model_path: str, device: str = "cuda"):
    """完全复刻单文件版本的模型加载逻辑"""
    extractor = get_DeepPrint_TexMinu(8000, 256)
    ckpt = torch.load(model_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    md = extractor.model.state_dict()
    md.update({k: v for k, v in sd.items() if k in md and hasattr(v, "shape") and v.shape == md[k].shape})
    extractor.model.load_state_dict(md)

    extractor.model.to(device)
    return extractor.model

def disable_dropout(model: torch.nn.Module):
    """完全复刻单文件版本的dropout禁用逻辑"""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()

def extract_minutiae_points(minu_map, img_gray, vis_img_u8, threshold=0.45, nms_size=5):
    """
    修复对齐问题：细节点坐标映射到可视化图像（vis_img_u8）尺寸，而非原始图像
    :param minu_map: 细节点图 [6, H, W]
    :param img_gray: 原始灰度图（用于生成掩码）
    :param vis_img_u8: 可视化用的图像（x_u8），用于坐标映射
    :param threshold: 响应值阈值
    :param nms_size: 非极大值抑制窗口大小
    :return: 细节点坐标列表 [(x1,y1), (x2,y2), ...]（匹配vis_img_u8尺寸）
    """
    # 1. 生成指纹区域掩码（和单文件版本一致）
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.resize(binary, (minu_map.shape[2], minu_map.shape[1])) > 0

    # 2. 计算通道维度最大值（响应值）（和单文件版本一致）
    max_val_map = np.max(minu_map, axis=0)  # [H, W]

    # 3. 非极大值抑制（和单文件版本一致）
    local_max = ndimage.maximum_filter(max_val_map, size=nms_size) == max_val_map

    # 4. 组合过滤条件（和单文件版本一致）
    final_mask = local_max & (max_val_map > threshold) & mask

    # 5. 获取细节点坐标（修复对齐：映射到可视化图像尺寸）
    coords = np.argwhere(final_mask)  # [N, 2] 格式: (y, x)
    
    # 尺寸映射参数（关键修复：匹配可视化图像vis_img_u8）
    h_minu, w_minu = minu_map.shape[1:]  # 细节点图尺寸
    h_vis, w_vis = vis_img_u8.shape      # 可视化图像尺寸
    
    minutiae_pts = []
    for y, x in coords:
        # 坐标映射到可视化图像尺寸（而非原始图像）
        x_vis = x * (w_vis / w_minu)
        y_vis = y * (h_vis / h_minu)
        minutiae_pts.append((int(x_vis), int(y_vis)))

    return minutiae_pts

def save_minutiae_visualization(img_u8, minutiae_pts, save_path):
    """完全复刻单文件版本的细节点可视化保存逻辑"""
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

def process_single_image(img_path, model, out_dir, args, loader):
    """处理单张图片，完全复刻单文件逻辑，仅输出指定两个文件"""
    try:
        # 1. 加载并预处理图片（和单文件版本完全一致）
        x, x_u8, img_original = loader(img_path)

        # 2. 构造模型输入张量（和单文件版本完全一致）
        if torch.is_tensor(x):
            xt = x
        else:
            xt = torch.from_numpy(np.asarray(x))

        if xt.ndim == 2:
            xt = xt.unsqueeze(0)  # [1,H,W]
        if xt.ndim == 3:
            xt = xt.unsqueeze(0)  # [B=1,C=1,H,W]

        # 补丁：避免batch维度被挤掉（和单文件版本完全一致）
        if xt.shape[0] == 1:
            xt = xt.repeat(2, 1, 1, 1)  # [2,1,H,W]

        xt = xt.to(device=args.device, dtype=torch.float32)

        # 3. 前向传播（和单文件版本完全一致）
        with torch.no_grad():
            out = model(xt)

        if not hasattr(out, "minutia_maps") or out.minutia_maps is None:
            raise RuntimeError(
                "Model output has no minutia_maps. "
                "This usually means you are not using TexMinu model OR forward did not enter training-output branch."
            )

        minu_tensor = out.minutia_maps  # [B,6,128,128]
        minu = minu_tensor[0].detach().float().cpu().numpy()  # [6,128,128]

        # 4. 提取细节点（修复对齐：传入可视化图像x_u8）
        minutiae_pts = extract_minutiae_points(
            minu_map=minu,
            img_gray=img_original,
            vis_img_u8=x_u8,  # 关键：传入可视化图像用于坐标映射
            threshold=args.minu_threshold,
            nms_size=args.nms_size
        )

        # 5. 生成并保存叠加原图可视化（完全复刻单文件版本逻辑）
        merged = np.max(minu, axis=0)
        merged = (merged - merged.min()) / (merged.max() - merged.min() + 1e-6)
        merged_u8 = (merged * 255).astype(np.uint8)
        heat_448 = cv2.resize(merged_u8, (x_u8.shape[1], x_u8.shape[0]), interpolation=cv2.INTER_CUBIC)
        heat_color = cv2.applyColorMap(heat_448, cv2.COLORMAP_JET)
        base_bgr = cv2.cvtColor(x_u8, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(base_bgr, 0.65, heat_color, 0.35, 0.0)
        overlay_path = os.path.join(out_dir, "overlay.png")
        cv2.imwrite(overlay_path, overlay)

        # 6. 保存细节点可视化图（完全复刻单文件版本逻辑）
        minutiae_vis_path = os.path.join(out_dir, "minutiae_points.png")
        save_minutiae_visualization(x_u8, minutiae_pts, minutiae_vis_path)

        print(f"处理完成: {os.path.basename(img_path)} | 细节点数量: {len(minutiae_pts)}")
        return True

    except Exception as e:
        print(f"处理失败: {os.path.basename(img_path)} | 错误: {str(e)}")
        return False

def main():
    ap = argparse.ArgumentParser()
    # 默认参数和单文件版本对齐
    ap.add_argument("--model", default="./ssh/example-model/best_model.pyt", help="模型文件路径")
    ap.add_argument("--indir", default=r"D:\AAAYan\ZhiWen\FixLength\fixed-length-fingerprint\ssh\data\test", help="输入图片文件夹路径（文件夹A）")
    ap.add_argument("--outdir", default="./ssh/minutia_output", help="批处理输出根目录")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    
    # 细节点检测参数（和单文件版本对齐）
    ap.add_argument("--minu-threshold", type=float, default=0.05, help="Minutiae response threshold")
    ap.add_argument("--nms-size", type=int, default=8.5, help="NMS window size for minutiae")

    # 保留兼容参数（和单文件版本完全一致）
    ap.add_argument("--crop-size", type=int, default=400)
    ap.add_argument("--angle", type=float, default=0.0)
    ap.add_argument("--roi-mode", default="NONE")
    ap.add_argument("--mask-apply-mode", default="white")
    ap.add_argument("--dpi", type=int, default=500)

    args = ap.parse_args()
    
    # 创建输出根目录
    os.makedirs(args.outdir, exist_ok=True)

    # 设备兼容（和单文件版本一致）
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print(f"CUDA不可用，自动切换到CPU模式")

    # 1. 加载模型（完全复刻单文件版本逻辑）
    print(f"加载模型: {args.model}")
    model = load_model(args.model, device=device)
    model.train()
    disable_dropout(model)

    # 2. 创建加载器（完全复刻单文件版本参数）
    loader = make_loader(
        crop_size=args.crop_size,
        angle=args.angle,
        roi_mode=args.roi_mode,
        mask_apply_mode=args.mask_apply_mode,
        dpi=args.dpi,
    )

    # 3. 遍历输入文件夹下的所有图片
    img_files = [f for f in os.listdir(args.indir) if f.lower().endswith(SUPPORTED_FORMATS)]
    if not img_files:
        print(f"输入文件夹 {args.indir} 中未找到支持的图片文件（{SUPPORTED_FORMATS}）")
        return

    print(f"\n开始批处理，共 {len(img_files)} 张图片")
    success_count = 0
    
    for img_name in img_files:
        # 为每张图片创建独立的输出子文件夹（避免文件覆盖）
        img_base = os.path.splitext(img_name)[0]
        img_out_dir = os.path.join(args.outdir, img_base)
        os.makedirs(img_out_dir, exist_ok=True)

        # 处理单张图片
        img_path = os.path.join(args.indir, img_name)
        if process_single_image(img_path, model, img_out_dir, args, loader):
            success_count += 1

    # 4. 输出批处理统计结果
    print(f"\n批处理完成 | 成功: {success_count} / 总数: {len(img_files)}")
    print(f"所有输出文件保存在: {args.outdir}")
    print(f"  每个图片的输出包含：overlay.png（叠加热力图）、minutiae_points.png（细节点可视化）")

if __name__ == "__main__":
    main()