# io.py
# -*- coding: utf-8 -*-
"""
数据读取与规范化工具（增强版，吸收 baseline Notebook 的习惯）：
- 统一 month / sector 的解析与格式化（字符串月份转 Timestamp）
- 从 id 拆解 month/sector（test.csv 用）
- 构造 Kaggle 提交 id： "YYYY Mon_sector n"
- 读取 train/ 多表并按 (month, sector) 合并
- 便捷的 target 提取与透视（time × sector），并可与 test 的 sector 对齐
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Kaggle 要求的英文月份缩写（固定映射，避免系统 locale 差异）
_EN_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------
#  Month / Sector 规范
# --------------------
def parse_month_col(df: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    """
    将 month 列解析为 pandas.Timestamp（对齐到当月第一天）。
    支持 "2019 Jan"、"2019-01"、"2019/01"、datetime 等常见格式。
    """
    if month_col not in df.columns:
        raise KeyError(f"month_col '{month_col}' not in dataframe")

    m = df[month_col]

    if np.issubdtype(m.dtype, np.datetime64):
        month_ts = m.dt.to_period("M").dt.to_timestamp()
    else:
        # 优先尝试 pandas 自动解析
        try:
            month_ts = pd.to_datetime(m, errors="raise")
            month_ts = month_ts.dt.to_period("M").dt.to_timestamp()
        except Exception:
            # 兜底：支持 "YYYY Mon" 与 "YYYY-MM"/"YYYY/MM"/"YYYY.M"
            def _parse_one(x: str):
                s = str(x).strip()
                mm = re.match(r"^(\d{4})\s+([A-Za-z]{3})$", s)
                if mm:
                    year = int(mm.group(1))
                    mon = _EN_MONTH_ABBR.index(mm.group(2).title()) + 1
                    return pd.Timestamp(year=year, month=mon, day=1)
                mm = re.match(r"^(\d{4})[-/\.](\d{1,2})$", s)
                if mm:
                    year = int(mm.group(1))
                    mon = int(mm.group(2))
                    return pd.Timestamp(year=year, month=mon, day=1)
                raise ValueError(f"Unknown month format: {x!r}")
            month_ts = m.map(_parse_one).astype("datetime64[ns]")

    out = df.copy()
    out[month_col] = month_ts
    return out


def normalize_sector_cols(
    df: pd.DataFrame,
    sector_col: str = "sector",
    out_int_col: str = "sector_id",
    keep_text_col: bool = True,
) -> pd.DataFrame:
    """
    将 sector 列标准化，抽取其中的整数 ID 到 `sector_id`。
    示例："sector 3" -> 3；5 -> 5。
    """
    if sector_col not in df.columns:
        raise KeyError(f"sector_col '{sector_col}' not in dataframe")

    def _to_int(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, np.integer)):
            return int(x)
        s = str(x)
        mm = re.search(r"(\d+)", s)
        if not mm:
            raise ValueError(f"Cannot extract sector id from {x!r}")
        return int(mm.group(1))

    out = df.copy()
    out[out_int_col] = out[sector_col].map(_to_int).astype("Int64")
    if not keep_text_col:
        out = out.drop(columns=[sector_col])
    return out


def build_id_col(
    df: pd.DataFrame,
    month_col: str = "month",
    sector_id_col: str = "sector_id",
    out_col: str = "id",
) -> pd.DataFrame:
    """
    根据 month 与 sector_id 构造竞赛要求的 id：
    形如 "2024 Aug_sector 1"
    """
    if month_col not in df.columns or sector_id_col not in df.columns:
        raise KeyError("month_col and sector_id_col must exist before build_id_col")

    months = pd.to_datetime(df[month_col]).dt
    year = months.year.astype(str)
    mon = months.month.apply(lambda m: _EN_MONTH_ABBR[m - 1])
    sector = df[sector_id_col].astype("Int64").astype(str)

    out = df.copy()
    out[out_col] = year + " " + mon + "_sector " + sector
    return out


def split_test_id(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """
    从 test.csv 的 id 拆解出 month/sector（文本形态），并追加标准化的 month/sector_id。
    """
    if id_col not in df.columns:
        raise KeyError(f"id_col '{id_col}' not in dataframe")
    out = df.copy()
    parts = out[id_col].str.split("_", n=1, expand=True)
    out["month"] = parts[0]
    out["sector"] = parts[1]
    out = parse_month_col(out, "month")
    out = normalize_sector_cols(out, "sector", out_int_col="sector_id", keep_text_col=True)
    # 重新生成 id（避免大小写或空白导致的不一致）
    out = build_id_col(out, "month", "sector_id", "id")
    return out


# --------------------
#  读取 / 合并 / 提取
# --------------------
@dataclass
class ReadConfig:
    """读取与合并配置"""
    train_dir: str                        # e.g. 'data/train'
    test_path: str                        # e.g. 'data/test.csv'
    base_table: str = "new_house_transactions.csv"  # 作为合并基表（含目标）
    month_col: str = "month"
    sector_col: str = "sector"
    target_col: str = "amount_new_house_transactions"
    ignore_files: Tuple[str, ...] = tuple()         # 可忽略某些表


def list_train_files(train_dir: str) -> List[str]:
    """列出 train_dir 下一级目录内的 CSV 文件"""
    fs = []
    for fn in os.listdir(train_dir):
        if fn.lower().endswith(".csv"):
            fs.append(os.path.join(train_dir, fn))
    return sorted(fs)


def read_csv(path: str) -> pd.DataFrame:
    """统一读取 CSV"""
    return pd.read_csv(path)


def _safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: Tuple[str, str],
    suffix: str,
) -> pd.DataFrame:
    """
    安全合并：避免列名冲突；右表除键外的列统一加后缀再合并。
    """
    k1, k2 = keys
    right_rename = {c: f"{c}{suffix}" for c in right.columns if c not in (k1, k2)}
    right = right.rename(columns=right_rename)
    return left.merge(right, on=[k1, k2], how="left")


def read_and_merge_train(config: ReadConfig) -> pd.DataFrame:
    """
    读取 train/ 下多表，按照 (month, sector) 合并为一个 DataFrame。
    - 基表默认使用 new_house_transactions.csv（含 target）
    - 其它表以文件名作为后缀，避免字段冲突
    """
    month_col, sector_col = config.month_col, config.sector_col
    train_files = [p for p in list_train_files(config.train_dir) if os.path.basename(p) not in config.ignore_files]
    if not train_files:
        raise FileNotFoundError(f"No csv files found in {config.train_dir}")

    # 读取并预处理所有表
    tables: Dict[str, pd.DataFrame] = {}
    for p in train_files:
        key = os.path.basename(p)
        df = read_csv(p)
        df = parse_month_col(df, month_col)
        df = normalize_sector_cols(df, sector_col, out_int_col="sector_id", keep_text_col=True)
        tables[key] = df

    if config.base_table not in tables:
        raise FileNotFoundError(
            f"Base table '{config.base_table}' not found. Found: {list(tables.keys())}"
        )

    base = tables[config.base_table].copy()

    # 依次合并其他表
    for name, df in tables.items():
        if name == config.base_table:
            continue
        suffix = "::" + name.replace(".csv", "")
        base = _safe_merge(base, df, keys=(month_col, sector_col), suffix=suffix)

    # sector_id 兜底
    if "sector_id" not in base.columns:
        base = normalize_sector_cols(base, sector_col, out_int_col="sector_id", keep_text_col=True)

    # 构造 id
    base = build_id_col(base, month_col=month_col, sector_id_col="sector_id", out_col="id")
    return memory_optimize(base)


def read_test(test_path: str) -> pd.DataFrame:
    """
    读取 test.csv，拆解 id 并规范化。
    """
    test = read_csv(test_path)
    test = split_test_id(test, id_col="id")
    return memory_optimize(test)


def memory_optimize(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    下采样数值类型以减少内存占用；字符串保持不变。
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_integer_dtype(s):
            out[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            out[col] = pd.to_numeric(s, downcast="float")
    if verbose:
        print(f"[memory_optimize] reduced to {out.memory_usage(deep=True).sum()/1024**2:.2f} MB")
    return out


# --------------------
#  透视 / 对齐工具
# --------------------
def make_target_pivot(
    df: pd.DataFrame,
    target_col: str = "amount_new_house_transactions",
    month_col: str = "month",
    sector_id_col: str = "sector_id",
    test_sectors: Optional[Iterable[int]] = None,
    fill_missing_zero: bool = True,
) -> pd.DataFrame:
    """
    将 (month, sector_id, target) 透视为 time×sector 的矩阵（行：月份，列：sector）。
    若 test_sectors 给出，则保证所有测试 sector 列都存在（缺失用 0 填充）。
    """
    tbl = df[[month_col, sector_id_col, target_col]].copy()
    tbl = tbl.pivot_table(index=month_col, columns=sector_id_col, values=target_col, aggfunc="sum")
    tbl = tbl.sort_index()

    if fill_missing_zero:
        tbl = tbl.fillna(0)

    if test_sectors is not None:
        # 确保列齐全
        test_sectors = list(sorted(set(int(s) for s in test_sectors)))
        for sid in test_sectors:
            if sid not in tbl.columns:
                tbl[sid] = 0
        tbl = tbl.reindex(columns=sorted(tbl.columns))  # 列按编号排序

    return tbl


def month_to_int_index(month_series: pd.Series) -> pd.Series:
    """
    将 Timestamp 的月份序列映射为从 0 开始的整数序号（按时间排序）。
    用于在需要“0..T”的 baseline 逻辑时复用（比如几何平均窗口）。
    """
    uniq = pd.Series(pd.to_datetime(month_series).dt.to_period("M").unique()).sort_values()
    mapping = {p: i for i, p in enumerate(uniq)}
    return pd.to_datetime(month_series).dt.to_period("M").map(mapping).astype(int)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
