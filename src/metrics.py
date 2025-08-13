# metrics.py
# -*- coding: utf-8 -*-
"""
竞赛自定义评估：Two-stage Scaled MAPE（与 baseline Notebook 一致）
- Stage 1：若 APE>1 的占比 > 0.3 -> score=0
- Stage 2：在 APE<=1 的集合 D 上计算平均，并按 D 占比缩放：score = 1 - mean(D) / (|D|/n)
- y_true==0：若 y_pred==0 -> APE=0；否则 APE=inf（促使 Stage1 触发）
"""

from __future__ import annotations
import numpy as np


def absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ape = np.empty_like(y_true, dtype=float)
    zero_mask = (y_true == 0)

    ape[zero_mask] = np.where(np.isclose(y_pred[zero_mask], 0.0), 0.0, np.inf)
    denom = np.maximum(np.abs(y_true[~zero_mask]), eps)
    ape[~zero_mask] = np.abs(y_pred[~zero_mask] - y_true[~zero_mask]) / denom
    return ape


def competition_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ape = absolute_percentage_error(y_true, y_pred)

    if np.mean(ape > 1.0) > 0.3:
        return 0.0

    D = ape[np.isfinite(ape) & (ape <= 1.0)]
    if D.size == 0:
        return 0.0
    p = D.size / ape.size
    return float(1.0 - D.mean() / p)

import numpy as np

def custom_score(y_true, y_pred, eps=1e-12):
    """
    题目给出的两阶段评估指标。

    参数
    ----
    y_true : array-like, shape (n_samples,)
        真实值
    y_pred : array-like, shape (n_samples,)
        预测值
    eps : float, optional
        防止除以 0 的小量，默认 1e-12

    返回
    ----
    float
        最终得分，范围 [0, 1]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        return 0.0

    # 绝对百分比误差
    ape = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))  # 避免 /0

    # 第一阶段：超过 100% 的样本比例
    bad_rate = np.mean(ape > 1.0)
    if bad_rate > 0.30:
        return 0.0

    # 第二阶段：只保留 ape <= 1.0 的样本
    mask = ape <= 1.0
    good_ape = ape[mask]

    if good_ape.size == 0:           # 极端：没有样本满足条件
        return 0.0

    # MAPE（仅 good_ape）
    mape = np.mean(good_ape)

    # 分数
    fraction = good_ape.size / y_true.size
    scaled_mape = mape / (fraction + eps)
    score = max(0.0, 1.0 - scaled_mape)
    return score

