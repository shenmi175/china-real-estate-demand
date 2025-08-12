# -*- coding: utf-8 -*-
"""
One-click baseline runner for the Kaggle 'China Real Estate Demand Prediction'
Usage examples:
  # 1) 启发式基线（默认：几何均值，t1=6, t2=6），并做CV评估
  python scripts/run_baseline.py --train_dir data/train --test_csv data/test.csv \
      --out submissions/sub_heuristic.csv --cv

  # 2) 普通算术均值的启发式，窗口改为 t1=12, t2=3
  python scripts/run_baseline.py --train_dir data/train --test_csv data/test.csv \
      --out submissions/sub_heuristic_mean.csv --method mean --t1 12 --t2 3

  # 3) LightGBM 基线（使用 src/features 的默认配置）
  python scripts/run_baseline.py --model lgbm --train_dir data/train --test_csv data/test.csv \
      --out submissions/sub_lgbm.csv
"""

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

# 让脚本能找到 src/ 包
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io import (
    ReadConfig,
    read_and_merge_train,
    read_test,
    make_target_pivot,
    build_id_col,
)
from src.metrics import competition_score
from src.model_lgbm import (
    HeuristicConfig,
    baseline_cv_score,
    baseline_predict_future,
    TrainConfig,
    cross_validate_time,
    fit_full,
    predict,
    export_submission,
)
from src.features import build_features
from src.cv import ExpandingWindowConfig, ExpandingWindowSplitter


def _heuristic_predict_submit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    out_path: str,
    t1: int = 6,
    t2: int = 6,
    method: str = "geomean",
    do_cv: bool = False,
):
    """启发式基线：基于 train 透视表逐月外推，生成 test 对应月份与 sector 的预测并导出。"""
    # 训练期的 time×sector 透视表（缺失填0，列齐全到 test 的 sector）
    test_sectors = test_df["sector_id"].unique().tolist()
    pivot = make_target_pivot(
        train_df,
        target_col=target_col,
        month_col="month",
        sector_id_col="sector_id",
        test_sectors=test_sectors,
        fill_missing_zero=True,
    )

    if do_cv:
        overall, per_folds = baseline_cv_score(
            pivot,
            HeuristicConfig(t1=t1, t2=t2, method=method),
            n_splits=4,
            valid_size_months=12,
        )
        print(f"[Heuristic CV] overall={overall:.5f} | folds={', '.join(f'{s:.5f}' for s in per_folds)}")

    # 需要预测的所有 test 月份（升序，Timestamp）
    test_months = pd.to_datetime(test_df["month"]).dt.to_period("M").dt.to_timestamp().sort_values().unique()

    # 递推式逐月产生预测行（如果 test 包含历史月，也用“历史到该月之前”的窗口生成）
    hist = pivot.copy()
    pred_rows = {}
    for m in test_months:
        # 以“严格小于 m 的历史”作为窗口
        hist_until_m = hist.loc[hist.index < m]
        if len(hist_until_m) == 0:
            # 无历史：全部置 0
            row = {sid: 0.0 for sid in hist.columns}
        else:
            cfg = HeuristicConfig(t1=t1, t2=t2, method=method)
            # 最近 t2 个月是否出现 0
            t2_eff = min(cfg.t2, len(hist_until_m))
            zero_mask = (hist_until_m.tail(t2_eff).min(axis=0) == 0)
            row = {}
            for sid, series in hist_until_m.items():
                if zero_mask[sid]:
                    row[sid] = 0.0
                else:
                    k = min(cfg.t1, len(hist_until_m))
                    vec = series.values[-k:]
                    if cfg.method == "geomean":
                        row[sid] = float(np.expm1(np.mean(np.log1p(vec))))
                    else:
                        row[sid] = float(np.mean(vec))
        # 记录并将该月“追加到历史”，用于预测更后续的月份
        pred_rows[m] = row
        hist = pd.concat([hist, pd.DataFrame([row], index=[m])])

    pred_df = pd.DataFrame.from_dict(pred_rows, orient="index").sort_index()
    pred_df.index.name = "month"

    # 按 test 的行序取值并导出
    # 先做一个 (month, sector_id) -> 预测 的快速索引
    lookup = pred_df.stack()
    lookup.index = lookup.index.set_names(["month", "sector_id"])
    pred_map = lookup.to_dict()

    preds = []
    miss = 0
    for m, sid in zip(test_df["month"], test_df["sector_id"]):
        key = (pd.to_datetime(m).to_period("M").to_timestamp(), int(sid))
        if key in pred_map:
            preds.append(pred_map[key])
        else:
            preds.append(0.0)
            miss += 1
    if miss:
        print(f"[warn] {miss} test rows were not matched in prediction map, filled by 0.")

    export_submission(test_df, np.array(preds, dtype=float), out_path=out_path)


def _lgbm_predict_submit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    out_path: str,
):
    """LightGBM 基线：构造特征→时间CV→全量拟合→预测 test→导出。"""
    # —— 1) 特征（这里给出保守的默认配置；字段不存在会被自动忽略）——
    lag_cfg = {
        target_col: [1, 3, 6, 12],
        "price_new_house_transactions": [1, 3, 6],
        "area_new_house_transactions": [1, 3, 6],
        "num_new_house_transactions": [1, 3, 6],
    }
    roll_cfg = {
        target_col: {"windows": [3, 6, 12], "stats": ["mean", "std"]},
        "price_new_house_transactions": {"windows": [3, 6], "stats": ["mean"]},
    }
    gr_cfg = {
        target_col: [1, 12],
    }
    log_cols = [target_col]

    train_fe, feat_cols = build_features(
        train_df,
        month_col="month",
        sector_id_col="sector_id",
        target_col=target_col,
        lag_cfg=lag_cfg,
        roll_cfg=roll_cfg,
        gr_cfg=gr_cfg,
        log_cols=log_cols,
        drop_list=[],
    )
    test_fe, _ = build_features(
        test_df,
        month_col="month",
        sector_id_col="sector_id",
        target_col=None,
        lag_cfg=lag_cfg,
        roll_cfg=roll_cfg,
        gr_cfg=gr_cfg,
        log_cols=log_cols,
        drop_list=[],
    )

    # —— 2) 时间序列 CV ——
    cv_cfg = ExpandingWindowConfig(
        time_col="month",
        n_splits=4,
        valid_size_months=12,
        gap_months=0,
        min_train_months=19,
    )
    splitter = ExpandingWindowSplitter(cv_cfg)
    splits = list(splitter.split(train_fe))

    train_cfg = TrainConfig(
        target_col=target_col,
        features=feat_cols,
        categorical_features=[],
        num_boost_round=4000,
        early_stopping_rounds=200,
    )
    cvres = cross_validate_time(train_fe, splits, train_cfg)
    oof_score = competition_score(train_fe[target_col].values, cvres.oof_pred)
    print(f"[LGBM] OOF competition_score={oof_score:.6f} | folds={', '.join(f'{s:.6f}' for s in cvres.fold_scores)}")

    # —— 3) 全量拟合 & 预测 ——
    final_model = fit_full(train_fe, train_cfg)
    test_pred = final_model.predict(test_fe[feat_cols], num_iteration=getattr(final_model, "best_iteration", None))

    export_submission(test_df, test_pred, out_path=out_path)


def main():
    ap = argparse.ArgumentParser(description="One-click baseline runner")
    ap.add_argument("--model", choices=["heuristic", "lgbm"], default="heuristic",
                    help="baseline type (default: heuristic)")
    ap.add_argument("--train_dir", type=str, required=True, help="path to data/train")
    ap.add_argument("--test_csv", type=str, required=True, help="path to data/test.csv")
    ap.add_argument("--out", type=str, required=True, help="output submission csv path")

    # heuristic params
    ap.add_argument("--t1", type=int, default=6, help="months window for mean/geomean")
    ap.add_argument("--t2", type=int, default=6, help="zero-gate window")
    ap.add_argument("--method", choices=["geomean", "mean"], default="geomean",
                    help="aggregation method for t1 window")
    ap.add_argument("--cv", action="store_true", help="run CV for heuristic baseline")

    args = ap.parse_args()

    # 读取合并 train（以 new_house_transactions.csv 为基表），以及 test
    cfg = ReadConfig(
        train_dir=args.train_dir,
        test_path=args.test_csv,
        base_table="new_house_transactions.csv",
        month_col="month",
        sector_col="sector",
        target_col="amount_new_house_transactions",
        ignore_files=tuple(),  # 可按需忽略某些表
    )
    train_df = read_and_merge_train(cfg)
    # 只保留最关键列（避免无关列干扰启发式或特征工程）
    keep_cols = [c for c in train_df.columns if c in
                 ["id", "month", "sector", "sector_id", cfg.target_col,
                  "price_new_house_transactions", "area_new_house_transactions", "num_new_house_transactions"]]
    train_df = train_df[keep_cols]

    test_df = read_test(args.test_csv)  # 含 month/sector_id/id

    # 路径准备
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.model == "heuristic":
        _heuristic_predict_submit(
            train_df=train_df,
            test_df=test_df,
            target_col=cfg.target_col,
            out_path=args.out,
            t1=args.t1,
            t2=args.t2,
            method=args.method,
            do_cv=bool(args.cv),
        )
    else:
        _lgbm_predict_submit(
            train_df=train_df,
            test_df=test_df,
            target_col=cfg.target_col,
            out_path=args.out,
        )

    print("[done] submission saved:", args.out)


if __name__ == "__main__":
    main()
