from .segmentation import *
from .orientations import *
from .frequencies import *
from .enhancement import *
import numpy as np
import cv2 as cv


_sufs_alg = None
_gmfs_alg = None
_snfoe_alg = None
_gbfoe_alg = None
_xsffe_alg = None
_snffe_alg = None
_snfen_alg = None
_gbfen_alg = None


def fingerprint_segmentation(fingerprint, dpi = 500, method = "SUFS"):
    """
    指纹分割的简单API

    参数
    ----------
    fingerprint : 包含指纹图像的Numpy数组(dtype: np.uint8)。
    dpi : 指纹分辨率。
    method : "SUFS" (需要 Keras) or "GMFS".

    返回值
    ----------    
    包含分割掩码的 NumPy 数组，其形状与输入指纹相同。
    """
    global _sufs_alg, _gmfs_alg
    if method == "SUFS":
        if _sufs_alg is None:
            _sufs_alg = Sufs()
        alg = _sufs_alg
    elif method == "GMFS":
        if _gmfs_alg is None:
            _gmfs_alg = Gmfs()
        alg = _gmfs_alg
    else:
        raise ValueError(f"Invalid method ({method})")
    alg.parameters.image_dpi = dpi
    return alg.run(fingerprint)


def orientation_field_estimation(fingerprint, segmentation_mask = None, dpi = 500, method = "SNFOE"):
    """
    指纹方向场估计的简单API

    参数
    ----------
    fingerprint:包含指纹图像的 NumPy 数组(数据类型:np. uint8)。

    segmentation_mask:包含分割掩码的 NumPy 数组(数据类型:np.uint8)。如果为 None,则使用整幅图像。

    dpi:指纹分辨率。

    method:"SNFOE"（需要 Keras)或 "GBFOE"。

    Returns
    ----------    
    A numpy array with the same shape of the input fingerprint containing the orientation at each pixel, 
    in radians.
    """
    global _snfoe_alg, _gbfoe_alg
    if method == "SNFOE":
        if _snfoe_alg is None:
            _snfoe_alg = Snfoe()
        alg = _snfoe_alg
    elif method == "GBFOE":
        if _gbfoe_alg is None:
            _gbfoe_alg = Gbfoe()
        alg = _gbfoe_alg
    else:
        raise ValueError(f"Invalid method ({method})")
    if segmentation_mask is None:
        segmentation_mask = np.full_like(fingerprint, 255)
    return alg.run(fingerprint, segmentation_mask, dpi)[0]


def frequency_estimation(fingerprint, orientation_field, segmentation_mask = None, dpi = 500, method = "SNFFE"):
    """
    指纹频率估计的简单API

    参数
    ----------
    fingerprint:包含指纹图像的 NumPy 数组(dtype: np.uint8)。

    orientation_field:包含每个像素脊线方向(弧度)的 NumPy 数组(dtype: np.float32)。

    segmentation_mask:包含分割掩码的 NumPy 数组(dtype: np.uint8)。如果为 None,则使用整幅图像。

    dpi:指纹分辨率。

    method:"SNFFE"(需要 Keras)或 "XSFFE"。

    返回值
    ----------    
    一个与输入指纹形状相同的 NumPy 数组，其中包含每个像素频率的倒数。
    """
    global _xsffe_alg, _snffe_alg
    if method == "SNFFE":
        if _snffe_alg is None:
            _snffe_alg = Snffe()
        alg = _snffe_alg
    elif method == "XSFFE":
        if _xsffe_alg is None:
            _xsffe_alg = Xsffe()
        alg = _xsffe_alg
    else:
        raise ValueError(f"Invalid method ({method})")
    if segmentation_mask is None:
        segmentation_mask = np.full_like(fingerprint, 255)
    return alg.run(fingerprint, segmentation_mask, orientation_field, dpi)


def fingerprint_enhancement(fingerprint, orientation_field, ridge_period_map, segmentation_mask = None, dpi = 500, method = "SNFEN"):
    """
    用于指纹增强的简单API

    参数
    ----------
    fingerprint:含指纹图像的 NumPy 数组(dtype: np.uint8)。

    orientation_field:包含每个像素脊线方向(弧度)的 NumPy 数组(dtype: np.float32)。

    ridge_period_map:包含每个像素脊线周期(频率的倒数)的 NumPy 数组。

    segmentation_mask:包含分割掩码的 NumPy 数组(dtype: np.uint8)。如果为 None,则使用整幅图像。

    dpi:指纹分辨率。

    method:"SNFEN"(需要 Keras)或 "GBFEN"。

    返回值
    ----------    
    增强后的图像，近似二值图像，脊线像素接近白色，谷部像素接近黑色。
    """
    global _snfen_alg, _gbfen_alg
    if method == "SNFEN":
        if _snfen_alg is None:
            _snfen_alg = Snfen()
        alg = _snfen_alg
    elif method == "GBFEN":
        if _gbfen_alg is None:
            _gbfen_alg = Gbfen()
        alg = _gbfen_alg
    else:
        raise ValueError(f"Invalid method ({method})")
    if segmentation_mask is None:
        segmentation_mask = np.full_like(fingerprint, 255)
    return alg.run(fingerprint, segmentation_mask, orientation_field, ridge_period_map, dpi)

def fingerprint_binarization(enhanced_gray: np.ndarray,
                             mask: np.ndarray | None = None,
                             method: str = "OTSU",
                             block_size: int = 31,
                             C: int = 5) -> np.ndarray:
    """
    将增强后的灰度图二值化（保持与 fingerprint_* 命名风格一致）
    enhanced_gray: uint8 灰度图（来自 fingerprint_enhancement 的输出）
    mask: 0/255 掩码；若提供，掩码外置零
    method: 'OTSU' 或 'ADAPTIVE'
    """
    assert enhanced_gray.ndim == 2, "fingerprint_binarization 需要灰度图"
    img = enhanced_gray
    if img.dtype != np.uint8:
        a, b = float(img.min()), float(img.max())
        img = ((img - a) * 255.0 / max(b - a, 1e-6)).astype(np.uint8)

    if method.upper() == "OTSU":
        _, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    elif method.upper() == "ADAPTIVE":
        if block_size % 2 == 0:
            block_size += 1
        binary = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY, block_size, C)
    else:
        raise ValueError("method 必须是 'OTSU' 或 'ADAPTIVE'")

    if mask is not None:
        binary = cv.bitwise_and(binary, mask)
    return binary.astype(np.uint8)


def fingerprint_thinning(binary: np.ndarray) -> np.ndarray:
    """
    Zhang–Suen 细化（骨架化），输入/输出均为 0/255 的 uint8 二值图
    """
    assert binary.ndim == 2 and binary.dtype == np.uint8
    img = (binary > 0).astype(np.uint8)
    h, w = img.shape
    changed = True
    while changed:
        changed = False
        # 子迭代 1
        to_del = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] == 0:
                    continue
                p2,p3,p4,p5,p6,p7,p8,p9 = (img[y-1,x], img[y-1,x+1], img[y,x+1], img[y+1,x+1],
                                           img[y+1,x], img[y+1,x-1], img[y,x-1], img[y-1,x-1])
                nz = p2+p3+p4+p5+p6+p7+p8+p9
                if nz < 2 or nz > 6:
                    continue
                seq = [p2,p3,p4,p5,p6,p7,p8,p9,p2]
                A = sum((seq[i] == 0 and seq[i+1] == 1) for i in range(8))
                if A != 1:
                    continue
                if p2 * p4 * p6 != 0:
                    continue
                if p4 * p6 * p8 != 0:
                    continue
                to_del.append((y, x))
        if to_del:
            for (y, x) in to_del:
                img[y, x] = 0
            changed = True

        # 子迭代 2
        to_del = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] == 0:
                    continue
                p2,p3,p4,p5,p6,p7,p8,p9 = (img[y-1,x], img[y-1,x+1], img[y,x+1], img[y+1,x+1],
                                           img[y+1,x], img[y+1,x-1], img[y,x-1], img[y-1,x-1])
                nz = p2+p3+p4+p5+p6+p7+p8+p9
                if nz < 2 or nz > 6:
                    continue
                seq = [p2,p3,p4,p5,p6,p7,p8,p9,p2]
                A = sum((seq[i] == 0 and seq[i+1] == 1) for i in range(8))
                if A != 1:
                    continue
                if p2 * p4 * p8 != 0:
                    continue
                if p2 * p6 * p8 != 0:
                    continue
                to_del.append((y, x))
        if to_del:
            for (y, x) in to_del:
                img[y, x] = 0
            changed = True

    return (img * 255).astype(np.uint8)