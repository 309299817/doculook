# Copyright (c) Opendatalab. All rights reserved.
import cv2
import numpy as np
from typing import Tuple, List
import math


def detect_text_orientation(image: np.ndarray) -> int:
    """
    检测图片中文本的主要方向
    
    Args:
        image: 输入图片 (numpy array)
    
    Returns:
        int: 需要旋转的角度 (0, 90, 180, 270)
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 使用边缘检测找到文本轮廓
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 使用霍夫变换检测直线
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is None:
        return 0
    
    angles = []
    for line in lines[:50]:  # 只取前50条线以提高性能
        rho, theta = line[0]
        angle = theta * 180 / np.pi
        
        # 将角度标准化到0-180度范围
        if angle > 90:
            angle = angle - 180
        angles.append(angle)
    
    if not angles:
        return 0
    
    # 统计角度分布，找到主要方向
    angle_hist = {}
    for angle in angles:
        # 将角度分组到10度的区间内
        bucket = round(angle / 10) * 10
        angle_hist[bucket] = angle_hist.get(bucket, 0) + 1
    
    # 找到最频繁的角度
    if not angle_hist:
        return 0
    
    dominant_angle = max(angle_hist, key=angle_hist.get)
    
    # 根据主要角度决定旋转角度
    if -15 <= dominant_angle <= 15:
        return 0  # 正常方向
    elif 75 <= abs(dominant_angle) <= 105:
        if dominant_angle > 0:
            return 270  # 需要逆时针旋转270度
        else:
            return 90   # 需要顺时针旋转90度
    elif abs(dominant_angle) >= 165 or abs(dominant_angle) <= 15:
        return 180  # 需要旋转180度
    else:
        return 0  # 其他情况不旋转


def detect_orientation_by_variance(image: np.ndarray) -> int:
    """
    通过投影方差检测文本方向
    这种方法基于文本行在正确方向时投影方差最大的原理
    
    Args:
        image: 输入图片
        
    Returns:
        int: 需要旋转的角度来纠正图片 (0, 90, 180, 270)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    variances = {}
    
    # 测试4个方向，看哪个方向的投影方差最大（即文本最正）
    test_rotations = {
        0: 0,      # 图片已经是正的，不需要旋转
        90: 270,   # 图片需要逆时针旋转270度来纠正
        180: 180,  # 图片需要旋转180度来纠正
        270: 90    # 图片需要顺时针旋转90度来纠正
    }
    
    for test_angle in [0, 90, 180, 270]:
        if test_angle == 0:
            rotated = binary
        else:
            rotated = rotate_image(binary, test_angle)
        
        # 计算水平投影的方差
        h_projection = np.sum(rotated, axis=1)
        variance = np.var(h_projection)
        variances[test_angle] = variance
    
    # 找到方差最大的测试角度，对应的纠正角度就是我们需要的
    best_test_angle = max(variances, key=variances.get)
    correction_angle = test_rotations[best_test_angle]
    
    return correction_angle


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """
    旋转图片
    
    Args:
        image: 输入图片
        angle: 旋转角度 (0, 90, 180, 270)
    
    Returns:
        np.ndarray: 旋转后的图片
    """
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # 对于其他角度使用仿射变换
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (width, height))


def auto_rotate_image(image: np.ndarray, method: str = "variance") -> Tuple[np.ndarray, int]:
    """
    自动检测并旋转图片到正确方向
    
    Args:
        image: 输入图片
        method: 检测方法 ("variance" 或 "hough")
    
    Returns:
        Tuple[np.ndarray, int]: (旋转后的图片, 旋转角度)
    """
    if method == "variance":
        rotation_angle = detect_orientation_by_variance(image)
    else:
        rotation_angle = detect_text_orientation(image)
    
    if rotation_angle == 0:
        return image, 0
    
    rotated_image = rotate_image(image, rotation_angle)
    return rotated_image, rotation_angle


def preprocess_image_orientation(image: np.ndarray, enable_auto_rotation: bool = True) -> Tuple[np.ndarray, dict]:
    """
    图片方向预处理
    
    Args:
        image: 输入图片
        enable_auto_rotation: 是否启用自动旋转
    
    Returns:
        Tuple[np.ndarray, dict]: (处理后的图片, 处理信息)
    """
    info = {
        "original_shape": image.shape,
        "rotation_applied": 0,
        "auto_rotation_enabled": enable_auto_rotation,
        "detection_methods": {}
    }
    
    if not enable_auto_rotation:
        return image, info
    
    try:
        # 尝试方差方法
        variance_angle = detect_orientation_by_variance(image)
        info["detection_methods"]["variance"] = variance_angle
        
        # 尝试霍夫变换方法
        hough_angle = detect_text_orientation(image)
        info["detection_methods"]["hough"] = hough_angle
        
        # 选择旋转角度：优先使用方差方法，如果没有检测到旋转则尝试霍夫变换
        rotation_angle = variance_angle if variance_angle != 0 else hough_angle
        
        info["rotation_applied"] = rotation_angle
        info["selected_method"] = "variance" if variance_angle != 0 else "hough"
        
        if rotation_angle != 0:
            rotated_image = rotate_image(image, rotation_angle)
            return rotated_image, info
        else:
            return image, info
        
    except Exception as e:
        info["error"] = str(e)
        return image, info


def batch_preprocess_images(images: List[np.ndarray], enable_auto_rotation: bool = True) -> List[Tuple[np.ndarray, dict]]:
    """
    批量处理图片方向
    
    Args:
        images: 图片列表
        enable_auto_rotation: 是否启用自动旋转
    
    Returns:
        List[Tuple[np.ndarray, dict]]: [(处理后的图片, 处理信息), ...]
    """
    results = []
    for image in images:
        processed_image, info = preprocess_image_orientation(image, enable_auto_rotation)
        results.append((processed_image, info))
    
    return results
