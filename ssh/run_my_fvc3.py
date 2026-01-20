import os
import sys
import numpy as np
import torch
import cv2
from sklearn.metrics import roc_curve
from pathlib import Path

# 设置项目路径
notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(notebook_dir)
sys.path.append(project_root)

# 导入必要模块
from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu, DeepPrintExtractor
from flx.scripts.generate_benchmarks import create_verification_benchmark
from flx.benchmarks.matchers import CosineSimilarityMatcher
from flx.data.embedding_loader import EmbeddingLoader
from flx.visualization.plot_DET_curve import plot_verification_results

# 项目原生类（无自定义类，避免导入错误）
from flx.data.dataset import Identifier, IdentifierSet, Dataset
from flx.data.image_loader import ImageLoader
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size


# ========================= 路径配置 =========================
MODEL_DIR = os.path.abspath("ssh/example-model")  # 模型目录
CUSTOM_DATA_DIR = r"D:\AAAYan\ZhiWen\FixLength\fixed-length-fingerprint\ssh\data\fingerprints\2Database"  # 你的指纹数据集路径
DET_FIGURE_PATH = "DET_curve_custom"  # DET曲线保存路径

# ========================= 评估参数 =========================
NUM_IMPRESSIONS_PER_SUBJECT = 4  # 每个手指的采集次数（你的数据集是4次）
TARGET_FAR_VALUES = [0.05, 0.01, 0.001, 0.0001, 0.00002, 0.00001]  # 目标FAR值


# ============================================================
# 第一步：实现适配你的数据集的ImageLoader（核心）
# ============================================================
class MyFingerprintLoader(ImageLoader):
    """适配你的指纹数据集的ImageLoader（兼容异常数据）"""
    @staticmethod
    def _extension() -> str:
        return ".bmp"  # 小写，配合lower()兼容大写

    @staticmethod
    def _file_to_id_fun(subdir: str, filename: str) -> Identifier:
        """解析文件名（兼容大写后缀+容错异常命名）"""
        filename_lower = filename.lower()
        name = filename_lower.replace(MyFingerprintLoader._extension(), "")
        parts = name.split("_")
        
        # 容错1：文件名格式错误（不足3部分）
        if len(parts) < 3:
            print(f"⚠️  跳过格式错误文件：{filename}（需x_x_x.bmp）")
            return None
        
        # 容错2：采集次数不是数字
        try:
            capture_id = int(parts[-1]) - 1
        except ValueError:
            print(f"⚠️  跳过采集次数错误文件：{filename}")
            return None
        
        # 解析finger_id并哈希
        finger_id = "_".join(parts[:-1])
        subject_id = hash(finger_id) % 1000000
        
        return Identifier(subject=subject_id, impression=capture_id)

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        """加载图片+预处理（保留原逻辑）"""
        img = cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️  无法加载图片，跳过：{filepath}")
            return None  # 返回None跳过损坏图片
        
        # 你的预处理逻辑（完全保留）
        crop_size_target = 400
        angle_target = 0
        
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
        
        # 适配模型输入尺寸并转Tensor
        img = pad_and_resize_to_deepprint_input_size(img, fill=1.0)
        if isinstance(img, np.ndarray):
            import torchvision.transforms.functional as VTF
            img = VTF.to_tensor(img)
        
        return img

    def __init__(self, root_dir: str):
        """初始化（移除断言，改为警告+详细统计）"""
        super().__init__(root_dir=Path(root_dir))
        total_samples = len(self.ids)
        total_finger_ids = self.ids.num_subjects
        
        # 打印核心统计
        print(f"📌 数据集验证：总图片数={total_samples}（预期6000），总指纹ID数={total_finger_ids}（预期1500）")
        
        # 警告而非断言
        if total_samples != 6000 or total_finger_ids != 1500:
            print(f"⚠️  警告：数据集数量不符！")
            print(f"   - 缺失图片数：{6000 - total_samples}")
            print(f"   - 缺失指纹ID数：{1500 - total_finger_ids}")
            
            # 统计采集次数不足4次的指纹ID
            subject_impression_count = {}
            for id_obj in self.ids:
                subject = id_obj.subject
                subject_impression_count[subject] = subject_impression_count.get(subject, 0) + 1
            
            insufficient_ids = [subj for subj, cnt in subject_impression_count.items() if cnt < 4]
            if insufficient_ids:
                print(f"⚠️  采集次数不足4次的指纹ID（前10个）：{insufficient_ids[:10]}")

# ============================================================
# 评估工具函数（保留原逻辑）
# ============================================================
def get_ffr_at_far(scores, labels, target_far):
    """根据目标FAR计算对应的FFR"""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr  # FFR = 1 - TPR
    idx = np.argmin(np.abs(fpr - target_far))
    return fnr[idx], thresholds[idx], fpr[idx]


# ============================================================
# 主流程（重构后）
# ============================================================
def main():
    # 1. 加载模型
    print("加载模型...")
    extractor: DeepPrintExtractor = get_DeepPrint_TexMinu(
        num_training_subjects=8000, 
        num_dims=256
    )
    extractor.load_best_model(MODEL_DIR)
    extractor.model.eval()  # 评估模式

    # 2. 加载自定义数据集（基于项目原生ImageLoader+Dataset）
    print("加载自定义数据集...")
    # 初始化自定义加载器
    custom_loader = MyFingerprintLoader(root_dir=CUSTOM_DATA_DIR)
    # 封装为项目原生Dataset（兼容后续评估逻辑）
    custom_dataset = Dataset(data_loader=custom_loader, identifier_set=custom_loader.ids)

    # 3. 提取特征嵌入（纹理+细节点）
    print("提取特征嵌入...")
    texture_embeddings, minutia_embeddings = extractor.extract(custom_dataset)
    embeddings = EmbeddingLoader.combine(texture_embeddings, minutia_embeddings)  # 合并特征

    # 可选：过滤采集次数不足4次的主体（避免基准创建报错）
    valid_subjects = []
    subject_impression_count = {}
    # 统计每个主体的采集次数
    for id_obj in custom_dataset.ids:
        subject = id_obj.subject
        subject_impression_count[subject] = subject_impression_count.get(subject, 0) + 1
    # 只保留采集次数≥4的主体
    valid_subjects = [subj for subj, cnt in subject_impression_count.items() if cnt >= 4]

    # 创建验证基准（用过滤后的有效主体）
    benchmark = create_verification_benchmark(
        subjects=valid_subjects,
        impressions_per_subject=list(range(NUM_IMPRESSIONS_PER_SUBJECT))
    )
    print(f"⚠️  过滤后有效主体数：{len(valid_subjects)}（原1499）")

    # 5. 余弦相似度匹配
    print("运行匹配测试...")
    matcher = CosineSimilarityMatcher(embeddings)
    results = benchmark.run(matcher)

    # 6. 整理分数和标签
    mated_scores = results.get_mated_scores()    # 同一手指的匹配分数（正样本）
    non_mated_scores = results.get_non_mated_scores()  # 不同手指的匹配分数（负样本）
    all_scores = np.concatenate([mated_scores, non_mated_scores])
    all_labels = np.concatenate([
        np.ones_like(mated_scores),   # 正样本标签=1
        np.zeros_like(non_mated_scores)  # 负样本标签=0
    ])

    # 7. 计算EER
    eer = results.get_equal_error_rate()
    print(f"\nEqual-Error-Rate (EER): {eer:.6f} ({eer*100:.4f}%)")

    # 8. 计算指定FAR对应的FFR
    print("\n===== FAR与对应的FFR =====")
    print(f"{'目标FAR':<12} {'实际FAR':<12} {'FFR':<12} {'阈值':<10}")
    print("-" * 50)
    for target_far in TARGET_FAR_VALUES:
        ffr, threshold, actual_far = get_ffr_at_far(all_scores, all_labels, target_far)
        print(f"{target_far:<12.6f} {actual_far:<12.6f} {ffr:<12.6f} {threshold:<10.4f}")

    # 9. 绘制DET曲线
    print("\n绘制DET曲线...")
    plot_verification_results(
        DET_FIGURE_PATH,
        results=[results],
        model_labels=["DeepPrint_TexMinu_Custom"],
        plot_title="Custom Fingerprint Dataset - Verification Performance"
    )
    print(f"DET曲线已保存至: {DET_FIGURE_PATH}")


if __name__ == "__main__":
    main()