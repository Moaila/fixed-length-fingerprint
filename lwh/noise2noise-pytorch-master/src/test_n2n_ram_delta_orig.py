# -*- coding: utf-8 -*-
"""
test_n2n_ram_delta_orig.py

测试“原始 Noise2Noise (U-Net) 模型”在单张图片推理时的 RAM 峰值增量。

流程：
1) 启动脚本，记录 RAM_base
2) 加载模型 + 预热一次（不计入单张图统计）
3) 再对同一张图进行一次推理：
   - 在推理前记录 RAM_before_infer
   - 推理过程中多次采样 RAM，记录一个 approx_peak
   - 推理后记录 RAM_after_infer
4) 输出：
   - RAM_delta_peak_from_start = RAM_peak - RAM_before_infer
   - RAM_delta_peak_from_end   = RAM_peak - RAM_after_infer

使用方式示例（CPU）：
  python test_n2n_ram_delta_orig.py \
      /home/tom/fingerprint-deepleaning/pyfing/tiqv_data/DB1_B/101_3.tif \
      --device cpu

默认 ckpt:
  ../ckpts/gaussian-clean/n2n-gaussian.pt
"""

import os
import sys
import time
import argparse
import threading

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F

# ===== psutil: 用于统计总内存(RAM) =====
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_cpu_mem_mb() -> float:
    """返回当前进程 RSS (MB)，若无 psutil 则返回 -1."""
    if not HAS_PSUTIL:
        return -1.0
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / 1024 / 1024


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))  # pyfing/noise2noise-pytorch-master


def load_n2n_unet_model(ckpt_path: str | None, device: torch.device) -> torch.nn.Module:
    """
    加载原始 Noise2Noise 仓库里的 U-Net 模型和权重。
    默认 in_channels=3。
    """
    if ckpt_path is None:
        ckpt_path = os.path.join(
            REPO_ROOT,
            "ckpts",
            "gaussian-clean",
            "n2n-gaussian.pt",
        )

    if THIS_DIR not in sys.path:
        sys.path.insert(0, THIS_DIR)

    from unet import UNet  # 原仓库 U-Net

    model = UNet(in_channels=3)
    state = torch.load(ckpt_path, map_location="cpu")

    # 兼容几种常见格式
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    print(f"[Orig-N2N] Loaded model: {ckpt_path}")
    print(f"[Orig-N2N] Device      : {device}")
    return model


@torch.no_grad()
def n2n_denoise(
    model: torch.nn.Module,
    device: torch.device,
    img_norm01: np.ndarray,
) -> np.ndarray:
    """
    img_norm01: (H, W) float32, [0,1]
    返回去噪结果 (H, W) float32, [0,1]
    """
    assert img_norm01.ndim == 2, "只支持灰度图 [H,W]"

    H, W = img_norm01.shape
    x = torch.from_numpy(img_norm01).float().unsqueeze(0).unsqueeze(0)
    x = x.repeat(1, 3, 1, 1).to(device)

    factor = 32
    pad_h = (factor - H % factor) % factor
    pad_w = (factor - W % factor) % factor
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    y = model(x)

    if y.ndim == 4:
        y = y[:, :, :H, :W]

    y = y.squeeze(0).detach().cpu().numpy()
    if y.ndim == 3:
        y = y.mean(axis=0)

    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    return y


def load_gray_norm01(path: str) -> np.ndarray:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="要去噪的灰度图路径")
    parser.add_argument("--ckpt", default=None, help="原始 Noise2Noise 模型 ckpt 路径")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="运行设备: auto=自动选择, cpu=强制CPU, cuda=强制GPU(若可用)"
    )
    parser.add_argument(
        "--sample-interval", type=float, default=0.005,
        help="推理时采样 RAM 的时间间隔(秒)，默认 5ms"
    )
    args = parser.parse_args()

    # ---------- 选择设备 ----------
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_cuda = (device.type == "cuda")

    img_path = os.path.abspath(args.image)
    img = load_gray_norm01(img_path)

    # 1) 脚本启动时总内存
    mem_base = get_cpu_mem_mb()

    # 2) 加载模型
    model = load_n2n_unet_model(args.ckpt, device=device)

    # 2.1) 预热一次（不计入单张图统计）
    _ = n2n_denoise(model, device, img)

    mem_after_model = get_cpu_mem_mb()

    # 3) 单张图 ΔRAM & 峰值 RAM 统计
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 在推理期间开一个 sampling 线程，不断读取 RAM，记录峰值
    ram_peak_holder = {"peak": get_cpu_mem_mb()}

    def _sampler(stop_event: threading.Event):
        while not stop_event.is_set():
            cur = get_cpu_mem_mb()
            if cur > ram_peak_holder["peak"]:
                ram_peak_holder["peak"] = cur
            time.sleep(args.sample_interval)

    stop_event = threading.Event()
    sampler_thread = threading.Thread(target=_sampler, args=(stop_event,))
    sampler_thread.daemon = True

    mem_before = get_cpu_mem_mb()
    if use_cuda:
        gpu_before = torch.cuda.memory_allocated() / 1024 / 1024
    else:
        gpu_before = 0.0

    # 开始采样
    sampler_thread.start()

    t0 = time.perf_counter()
    out = n2n_denoise(model, device, img)
    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # 停止采样
    stop_event.set()
    sampler_thread.join()

    mem_after = get_cpu_mem_mb()
    mem_peak = ram_peak_holder["peak"]

    if use_cuda:
        gpu_after = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        gpu_after = 0.0
        gpu_peak = 0.0

    # 4) 打印结果
    print(f"\n[Orig-N2N ΔRAM] image: {img_path}")
    print(f"  device                     : {device}")
    print(f"  time (single infer)        : {t1 - t0:.4f} s")

    if HAS_PSUTIL:
        print(f"  RAM_base                   : {mem_base:.1f} MB (脚本启动时)")
        print(f"  RAM_after_model            : {mem_after_model:.1f} MB (模型加载+预热后)")
        print(f"  RAM_before_infer           : {mem_before:.1f} MB (单张图推理前)")
        print(f"  RAM_after_infer            : {mem_after:.1f} MB (单张图推理后)")
        print(f"  RAM_peak_during_infer      : {mem_peak:.1f} MB (推理过程采样到的峰值)")
        print(f"  RAM_overhead(model)        : {mem_after_model - mem_base:.1f} MB (模型带来的净增加)")
        print(f"  RAM_delta_peak_from_start  : {mem_peak - mem_before:.1f} MB (峰值-推理开始)")
        print(f"  RAM_delta_peak_from_end    : {mem_peak - mem_after:.1f} MB (峰值-推理结束)")
    else:
        print("  (psutil 未安装，CPU 总内存无法统计)")

    if use_cuda:
        print(f"  GPU_mem_before             : {gpu_before:.1f} MB (allocated)")
        print(f"  GPU_mem_after              : {gpu_after:.1f} MB (allocated)")
        print(f"  GPU_mem_peak               : {gpu_peak:.1f} MB (peak allocated)")

    out_u8 = (out * 255.0 + 0.5).astype(np.uint8)
    out_save = os.path.splitext(img_path)[0] + "_n2n_orig.png"
    cv.imwrite(out_save, out_u8)
    print(f"  denoised saved             : {out_save}")


if __name__ == "__main__":
    main()
