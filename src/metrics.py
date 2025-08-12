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
