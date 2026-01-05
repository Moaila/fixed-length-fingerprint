import os
import sys
import argparse
import numpy as np
import cv2
import torch

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
    """
    轻量化预处理逻辑：
    - 读灰度图
    - CLAHE增强
    - 直接pad+resize到DeepPrint输入尺寸
    返回: 可调用的加载函数（移除错误的staticmethod装饰器）
    """
    # 移除 @staticmethod 装饰器！！！
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

        return x, x_u8

    return _load  # 返回普通函数，而非静态方法

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--outdir", default="./examples/out/minumap_vis")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

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

    # 2. 创建加载器（现在返回的是可调用函数）
    loader = make_loader(
        crop_size=args.crop_size,
        angle=args.angle,
        roi_mode=args.roi_mode,
        mask_apply_mode=args.mask_apply_mode,
        dpi=args.dpi,
    )
    # 调用加载器处理图片（现在可正常调用）
    x, x_u8 = loader(args.img)

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

    # 5. 保存各类输出
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

    # 8. 叠加原图可视化
    heat_448 = cv2.resize(merged_u8, (x_u8.shape[1], x_u8.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(heat_448, cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(x_u8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.65, heat_color, 0.35, 0.0)

    cv2.imwrite(os.path.join(args.outdir, "input_448.png"), x_u8)
    cv2.imwrite(os.path.join(args.outdir, "overlay.png"), overlay)

    print(" saved to:", args.outdir)
    print("Files: input_448.png, overlay.png, minu_merged.png, minu_ch0..5.png, minu_map_raw.npy")

if __name__ == "__main__":
    main()