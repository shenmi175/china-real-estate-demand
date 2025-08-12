# cv.py
# -*- coding: utf-8 -*-
"""
时间序列交叉验证分割器：
- ExpandingWindowSplitter：按月滚动的“扩展训练窗 + 固定验证窗”
- 兼容 baseline：可指定每折验证覆盖的月份数（如 12）
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Generator, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ExpandingWindowConfig:
    time_col: str = "month"
    n_splits: int = 4
    valid_size_months: int = 12
    gap_months: int = 0
    min_train_months: int = 19  # baseline 第一折：0..18 训练


class ExpandingWindowSplitter:
    def __init__(self, config: ExpandingWindowConfig):
        self.cfg = config

    def split(self, df: pd.DataFrame):
        time_col = self.cfg.time_col
        months = pd.to_datetime(df[time_col]).dt.to_period("M").astype(str)
        uniq_months = pd.Series(months.unique()).sort_values().tolist()

        folds = []
        train_end = self.cfg.min_train_months
        for _ in range(self.cfg.n_splits):
            valid_start = train_end
            valid_end = valid_start + self.cfg.valid_size_months
            if valid_end > len(uniq_months):
                break
            train_m = uniq_months[:train_end]
            valid_m = uniq_months[valid_start:valid_end]
            folds.append((train_m, valid_m))
            train_end = valid_end

        month_series = months.values
        for train_m, valid_m in folds:
            tr_mask = np.isin(month_series, train_m)
            va_mask = np.isin(month_series, valid_m)
            yield np.where(tr_mask)[0], np.where(va_mask)[0]
