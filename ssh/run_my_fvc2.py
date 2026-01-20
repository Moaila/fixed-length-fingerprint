import os
import sys
import numpy as np
import torch
from sklearn.metrics import roc_curve

"""
普通评估
"""
# 设置项目路径
notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(notebook_dir)
sys.path.append(project_root)

# 导入必要模块
from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu, DeepPrintExtractor
from flx.setup.datasets import get_fvc2004_db1a
from flx.scripts.generate_benchmarks import create_verification_benchmark
from flx.benchmarks.matchers import CosineSimilarityMatcher
from flx.data.embedding_loader import EmbeddingLoader
from flx.visualization.plot_DET_curve import plot_verification_results

# ========================= 路径配置 =========================
MODEL_DIR = os.path.abspath("example-model")  # 模型目录
FVC_DB1A_DIR = r"D:\AAAduYan\ZhiWen\Xin12_4\fixed-length-fingerprint-extractors\data\fingerprints\FVC2000\Dbs\Db1_a"  # 数据集路径
DET_FIGURE_PATH = "DET_curve"  # DET曲线保存路径

# ========================= 评估参数 =========================
NUM_IMPRESSIONS_PER_SUBJECT = 8  # 每个主体的样本数
# 需要计算的FAR值（False Acceptance Rate）
TARGET_FAR_VALUES = [0.05, 0.01, 0.001, 0.0001, 0.00002, 0.00001]

# ============================================================
#                     评估工具函数
# ============================================================

def get_ffr_at_far(scores, labels, target_far):
    """根据目标FAR计算对应的FFR"""
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr  # FFR = 1 - TPR
    
    # 找到最接近目标FAR的阈值
    idx = np.argmin(np.abs(fpr - target_far))
    return fnr[idx], thresholds[idx], fpr[idx]

def get_all_far_ffr(scores, labels):
    """获取完整的FAR-FFR曲线数据"""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    return fpr, fnr, thresholds

# ============================================================
#                     主流程
# ============================================================

def main():
    # 1. 加载模型
    print("加载模型...")
    extractor: DeepPrintExtractor = get_DeepPrint_TexMinu(
        num_training_subjects=8000, 
        num_dims=256
    )
    extractor.load_best_model(MODEL_DIR)
    extractor.model.eval()  # 设置为评估模式

    # 2. 加载数据集
    print("加载数据集...")
    fvc_dataset = get_fvc2004_db1a(FVC_DB1A_DIR)
    print(f"数据集加载完成: {fvc_dataset.num_subjects}个主体, {len(fvc_dataset)}个样本")

    # 3. 提取特征嵌入
    print("提取特征嵌入...")
    texture_embeddings, minutia_embeddings = extractor.extract(fvc_dataset)
    # 合并纹理和minutiae特征
    embeddings = EmbeddingLoader.combine(texture_embeddings, minutia_embeddings)

    # 4. 创建验证基准
    print("创建验证基准...")
    benchmark = create_verification_benchmark(
        subjects=list(range(fvc_dataset.num_subjects)),
        impressions_per_subject=list(range(NUM_IMPRESSIONS_PER_SUBJECT))
    )

    # 5. 运行匹配并获取结果
    print("运行匹配测试...")
    matcher = CosineSimilarityMatcher(embeddings)
    results = benchmark.run(matcher)

    # 6. 提取分数和标签用于计算
    mated_scores = results.get_mated_scores()  # 真实匹配分数（正样本）
    non_mated_scores = results.get_non_mated_scores()  # 虚假匹配分数（负样本）
    
    # 构建完整的分数和标签数组
    all_scores = np.concatenate([mated_scores, non_mated_scores])
    all_labels = np.concatenate([
        np.ones_like(mated_scores),  # 正样本标签为1
        np.zeros_like(non_mated_scores)  # 负样本标签为0
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
        print(
            f"{target_far:<12.6f} {actual_far:<12.6f} {ffr:<12.6f} {threshold:<10.4f}"
        )

    # 9. 绘制DET曲线
    print("\n绘制DET曲线...")
    plot_verification_results(
        DET_FIGURE_PATH,
        results=[results],
        model_labels=["DeepPrint_TexMinu"],
        plot_title="FVC2004 DB1A - Verification Performance"
    )
    print(f"DET曲线已保存至: {DET_FIGURE_PATH}")

if __name__ == "__main__":
    main()