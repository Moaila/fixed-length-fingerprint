import os
import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu
from flx.setup.datasets import get_fvc2004_db1a
from flx.data.image_loader import FVC2004Loader

# ========================= 配置项 =========================
MODEL_PATH = "./models/best_model.pyt"  # 你的模型路径
DATA_DIR   = "./data/fingerprints/test"     # 你的数据集路径
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备
INPUT_SIZE = 448  # 模型输入尺寸（根据你的实际情况调整，默认448）
OUTPUT_DIR = "./minutia_output/1"  # 结果保存目录

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
#                     核心工具：特征Hook + 可视化
# ============================================================
class FeatureHook:
    """用于捕获模型中间层输出的 Hook（仅保留核心功能）"""
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.output = None

    def hook_fn(self, module, input, output):
        self.output = output

    def close(self):
        self.hook.remove()

def save_and_show_minutia_map(extractor, ds, device=DEVICE):
    """
    核心功能：
    1. 提取6通道minutia_map并单独可视化
    2. 生成通道最大值融合图
    3. 将融合图叠加到原始指纹图像并保存（修复尺寸匹配问题）
    """
    # 定位 minutia_map 最后卷积层（生成6通道特征图的层）
    target_layer = extractor.model.minutia_map.features[3]
    hook = FeatureHook(target_layer)

    # 取数据集第一张图作为可视化样本
    data = ds[0]
    img_tensor = data[0] if isinstance(data, (tuple, list)) else data
    # 保存原始图像（用于后续叠加）
    if not torch.is_tensor(img_tensor):
        original_img_np = img_tensor.copy()  # 原始预处理后的图像（0-1浮点型）
        img_tensor = torch.from_numpy(img_tensor)
    else:
        original_img_np = img_tensor.cpu().numpy()  # 转numpy
    
    # 构造模型输入（避免维度压缩，重复1次batch）
    img_input = img_tensor.unsqueeze(0).repeat(2, 1, 1, 1).to(device).float()

    print(f"📌 目标层: {target_layer}")
    print("🔄 切换模型为train模式以激活minutiae分支...")
    extractor.model.train()  # 必须train模式才能输出minutia_maps

    # 前向传播获取特征
    with torch.no_grad():
        try:
            extractor.model(img_input)
        except Exception as e:
            print(f"❌ 前向传播错误: {e}")
            hook.close()
            return

    # 检查是否成功捕获特征
    if hook.output is None:
        print("❌ 未捕获到minutiae特征图，请检查模型结构！")
        hook.close()
        return

    # 提取并处理特征图（取第一个样本的6通道特征）
    m_map = hook.output[0].cpu().detach().numpy()  # [6, 128, 128]
    hook.close()
    extractor.model.eval()  # 恢复eval模式

    # ===================== 1. 绘制6通道特征图（原有逻辑） =====================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("DeepPrint Minutia Map (6 Channels)", fontsize=16)
    for i in range(6):
        ax = axes[i // 3, i % 3]
        im = ax.imshow(m_map[i], cmap='gray_r')  # gray_r更清晰展示特征
        ax.set_title(f"Channel {i}", fontsize=12)
        ax.axis('off')  # 关闭坐标轴
        plt.colorbar(im, ax=ax, shrink=0.8)  # 颜色条
    
    # 保存6通道可视化图
    six_chan_path = os.path.join(OUTPUT_DIR, "minutia_map_6channels.png")
    plt.tight_layout()
    plt.savefig(six_chan_path, dpi=150, bbox_inches='tight')
    plt.close()  # 关闭画布释放内存
    print(f"✅ 6通道特征图已保存: {six_chan_path}")

    # ===================== 2. 生成通道最大值融合图 =====================
    print("🔧 生成通道最大值融合图...")
    # 沿通道维度取最大值（融合所有角度的特征点）
    merged = np.max(m_map, axis=0)  # [128, 128]
    # 归一化到0-1（消除数值范围差异）
    merged = (merged - merged.min()) / (merged.max() - merged.min() + 1e-6)
    # 转换为uint8（0-255）用于保存和叠加
    merged_u8 = (merged * 255).astype(np.uint8)

    # 保存融合图
    merged_path = os.path.join(OUTPUT_DIR, "minutia_merged.png")
    cv2.imwrite(merged_path, merged_u8)
    print(f"✅ 通道最大值融合图已保存: {merged_path}")

    # ===================== 3. 融合图叠加到原始指纹图像（修复尺寸匹配） =====================
    print("🎨 将融合图叠加到原始指纹图像...")
    # 步骤1：处理原始图像（标准化格式，确保尺寸/通道正确）
    # 原始图像是0-1浮点型，转255尺度
    original_img_u8 = (original_img_np * 255).astype(np.uint8)
    # 若原始图像是3维（1, H, W），压缩为2维
    if original_img_u8.ndim == 3:
        original_img_u8 = original_img_u8.squeeze(0)
    # 打印原始图像尺寸（调试用）
    print(f"   - 原始图像尺寸: {original_img_u8.shape}")

    # 步骤2：强制对齐尺寸（核心修复）
    # 获取原始图像的实际尺寸
    h, w = original_img_u8.shape[:2]
    # 将融合图缩放至原始图像的实际尺寸（而非固定INPUT_SIZE）
    merged_resized = cv2.resize(merged_u8, (w, h), interpolation=cv2.INTER_CUBIC)
    print(f"   - 融合图缩放后尺寸: {merged_resized.shape}")

    # 步骤3：转换为彩色热力图（确保3通道）
    heat_map = cv2.applyColorMap(merged_resized, cv2.COLORMAP_JET)
    # 确保热力图是3通道（防止特殊情况）
    if heat_map.ndim != 3 or heat_map.shape[2] != 3:
        heat_map = cv2.cvtColor(heat_map, cv2.COLOR_GRAY2BGR)
    print(f"   - 热力图尺寸/通道: {heat_map.shape}")

    # 步骤4：原始灰度图转彩色（确保3通道，与热力图匹配）
    if original_img_u8.ndim == 2:  # 单通道灰度图
        original_bgr = cv2.cvtColor(original_img_u8, cv2.COLOR_GRAY2BGR)
    else:  # 已为彩色图
        original_bgr = original_img_u8
    print(f"   - 原始图转彩色后尺寸/通道: {original_bgr.shape}")

    # 最终校验：确保两张图尺寸/通道完全一致
    if original_bgr.shape != heat_map.shape:
        print(f"⚠️  尺寸/通道不匹配，强制对齐: {original_bgr.shape} → {heat_map.shape}")
        # 终极兜底：缩放热力图到原始图尺寸
        heat_map = cv2.resize(heat_map, (original_bgr.shape[1], original_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 步骤5：图像叠加（原始图65% + 热力图35%，透明度可调）
    overlay = cv2.addWeighted(original_bgr, 0.65, heat_map, 0.35, 0.0)
    
    # 保存叠加图
    overlay_path = os.path.join(OUTPUT_DIR, "minutia_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"✅ 融合图叠加到原图已保存: {overlay_path}")

    # 额外：保存原始指纹图像（方便对比）
    original_path = os.path.join(OUTPUT_DIR, "original_fingerprint.png")
    cv2.imwrite(original_path, original_img_u8)
    print(f"✅ 原始指纹图像已保存: {original_path}")

    print("\n🎉 所有结果保存完成！输出目录：", OUTPUT_DIR)
    print("生成文件列表：")
    print(f"  - {six_chan_path}: 6通道minutia map单独可视化")
    print(f"  - {merged_path}: 6通道最大值融合图（128x128）")
    print(f"  - {original_path}: 原始预处理指纹图像（{h}x{w}）")
    print(f"  - {overlay_path}: 融合特征图叠加到原图（热力图）")

# ============================================================
#                   简化版图像预处理（仅用于可视化）
# ============================================================
def make_simple_loader():
    """
    极简预处理：仅保留可视化所需的基础处理
    核心修复：添加self参数，匹配类方法的调用规则
    """
    # 定义类方法格式的加载函数：第一个参数为self（类实例），第二个为filepath
    def _load(self, filepath):
        # 基础预处理：灰度读取 + CLAHE增强 + 缩放至模型输入尺寸
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"❌ 读取图片失败: {filepath}")
        
        # CLAHE增强（固定参数，保证可视化效果）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        # 缩放至DeepPrint输入尺寸（448x448）
        from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
        img_processed = pad_and_resize_to_deepprint_input_size(img, fill=1.0)
        return img_processed

    return _load

# ============================================================
#                       主流程（仅可视化）
# ============================================================
def main():
    """主函数：仅加载模型 + 可视化minutiae特征图 + 融合叠加"""
    # 1. 加载预训练模型
    print("📥 加载模型...")
    extractor = get_DeepPrint_TexMinu(8000, 256)  # 参数仅为构造模型，不影响可视化
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # 加载权重（匹配维度）
    md = extractor.model.state_dict()
    md.update({k: v for k, v in sd.items() if k in md and v.shape == md[k].shape})
    extractor.model.load_state_dict(md)
    extractor.model.to(DEVICE).eval()
    print("✅ 模型加载完成！")

    # 2. 配置数据集加载器（简化版预处理）
    print("\n📂 加载数据集...")
    # 替换FVC2004Loader的_load_image方法（类方法，需接收self参数）
    FVC2004Loader._load_image = make_simple_loader()
    ds = get_fvc2004_db1a(DATA_DIR)
    
    if len(ds) == 0:
        print("❌ 数据集为空，请检查DATA_DIR路径！")
        return
    print(f"✅ 数据集加载完成，共 {len(ds)} 个样本")

    # 3. 核心：可视化minutiae特征图 + 融合叠加
    print("\n🎨 开始可视化minutiae特征图...")
    save_and_show_minutia_map(extractor, ds)

if __name__ == "__main__":
    main()