# features.py
# -*- coding: utf-8 -*-
"""
时序与面板特征构造：
- 日期衍生（年、月、季度、年内序号）
- 分组滞后 / 滚动统计 / 增长率（严格避免使用未来信息）
- log1p 变换等
- 提供统一入口 build_features 便于配置化扩展
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def add_date_parts(
    df: pd.DataFrame,
    month_col: str = "month",
    prefix: str = "date_",
    drop_original: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    m = pd.to_datetime(out[month_col])
    out[f"{prefix}year"] = m.dt.year.astype(np.int16)
    out[f"{prefix}month"] = m.dt.month.astype(np.int8)
    out[f"{prefix}quarter"] = m.dt.quarter.astype(np.int8)
    out[f"{prefix}month_index"] = (out[f"{prefix}year"] - out[f"{prefix}year"].min()) * 12 + (out[f"{prefix}month"] - 1)
    if drop_original:
        out = out.drop(columns=[month_col])
    return out


def _group_sorted(df: pd.DataFrame, group_col: str, time_col: str):
    return df.sort_values([group_col, time_col]).groupby(group_col, sort=False, group_keys=False)


def add_group_lag(
    df: pd.DataFrame,
    cols: Sequence[str],
    lags: Sequence[int],
    group_col: str = "sector_id",
    time_col: str = "month",
    suffix: str = "lag",
) -> pd.DataFrame:
    out = df.copy()
    g = _group_sorted(out, group_col, time_col)
    for col in cols:
        for L in lags:
            out[f"{col}_{suffix}{L}"] = g[col].shift(L)
    return out


def add_group_rolling(
    df: pd.DataFrame,
    cols: Sequence[str],
    windows: Sequence[int],
    stats: Sequence[str] = ("mean", "std", "min", "max", "sum"),
    group_col: str = "sector_id",
    time_col: str = "month",
    min_periods: Optional[int] = None,
    center: bool = False,
    suffix: str = "roll",
) -> pd.DataFrame:
    out = df.copy()
    g = _group_sorted(out, group_col, time_col)

    for col in cols:
        for w in windows:
            r = g[col].shift(1).rolling(
                window=w,
                min_periods=min_periods if min_periods is not None else max(1, w // 2),
                center=center,
            )
            if "mean" in stats:
                out[f"{col}_{suffix}{w}_mean"] = r.mean()
            if "std" in stats:
                out[f"{col}_{suffix}{w}_std"] = r.std(ddof=0)
            if "min" in stats:
                out[f"{col}_{suffix}{w}_min"] = r.min()
            if "max" in stats:
                out[f"{col}_{suffix}{w}_max"] = r.max()
            if "sum" in stats:
                out[f"{col}_{suffix}{w}_sum"] = r.sum()
    return out


def add_group_growth(
    df: pd.DataFrame,
    cols: Sequence[str],
    periods: Sequence[int],
    group_col: str = "sector_id",
    time_col: str = "month",
    suffix: str = "gr",
) -> pd.DataFrame:
    out = df.copy()
    g = _group_sorted(out, group_col, time_col)
    for col in cols:
        for p in periods:
            prev = g[col].shift(p)
            out[f"{col}_{suffix}{p}"] = (out[col] - prev) / prev.replace(0, np.nan)
    return out


def add_log1p(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[f"{c}_log1p"] = np.log1p(out[c].clip(lower=0))
    return out


def drop_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    exist = [c for c in cols if c in df.columns]
    return df.drop(columns=exist)


def select_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    id_cols: Sequence[str] = ("id", "month", "sector", "sector_id"),
    extra_exclude: Sequence[str] = (),
) -> List[str]:
    exclude = set(id_cols) | {target_col} | set(extra_exclude)
    feats = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return feats


def build_features(
    df: pd.DataFrame,
    *,
    month_col: str = "month",
    sector_id_col: str = "sector_id",
    target_col: Optional[str] = None,
    lag_cfg: Dict[str, Sequence[int]] | None = None,
    roll_cfg: Dict[str, Dict[str, Sequence]] | None = None,
    gr_cfg: Dict[str, Sequence[int]] | None = None,
    log_cols: Sequence[str] | None = None,
    drop_list: Sequence[str] = (),
) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    out = add_date_parts(out, month_col)

    if lag_cfg:
        for col, lags in lag_cfg.items():
            if col in out.columns:
                out = add_group_lag(out, [col], lags, group_col=sector_id_col, time_col=month_col)

    if roll_cfg:
        for col, cfg in roll_cfg.items():
            if col in out.columns:
                windows: Sequence[int] = cfg.get("windows", [3])
                stats: Sequence[str] = cfg.get("stats", ("mean", "std"))
                out = add_group_rolling(
                    out, [col], windows, stats=stats, group_col=sector_id_col, time_col=month_col
                )

    if gr_cfg:
        for col, periods in gr_cfg.items():
            if col in out.columns:
                out = add_group_growth(out, [col], periods, group_col=sector_id_col, time_col=month_col)

    if log_cols:
        present = [c for c in log_cols if c in out.columns]
        out = add_log1p(out, present)

    if drop_list:
        out = drop_cols(out, drop_list)

    feat_cols = select_feature_columns(out, target_col or "__no_target__", extra_exclude=drop_list)
    return out, feat_cols
