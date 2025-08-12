# model_lgbm.py
# -*- coding: utf-8 -*-
"""
模型与基线：
- LightGBM 回归训练 / 时间CV / 预测 / 提交导出
- 简单启发式基线（来源于官方 baseline 思路）：按每个 sector
  取最近 t1 个月的（几何或算术）均值作为未来月份预测；若最近 t2 个月存在 0，则预测 0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from .metrics import competition_score



# =========================
# LightGBM 训练 / 推理模块
# =========================
@dataclass
class TrainConfig:
    target_col: str = "amount_new_house_transactions"
    features: Optional[List[str]] = None
    categorical_features: Sequence[str] = ()
    num_boost_round: int = 5000
    early_stopping_rounds: int = 200
    params: Dict = field(default_factory=lambda: dict(
        objective="mae",
        metric="l1",
        learning_rate=0.05,
        num_leaves=256,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=50,
        max_depth=-1,
        verbosity=-1,
        n_estimators=5000,
    ))
    seed: int = 42
    verbose_eval: int = 100


@dataclass
class CVResult:
    models: List[lgb.Booster]
    oof_pred: np.ndarray
    fold_scores: List[float]
    feature_importance: pd.DataFrame


def _lgb_dataset(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    cat_cols: Sequence[str] = (),
) -> lgb.Dataset:
    params = {}
    if cat_cols:
        params["categorical_feature"] = list(cat_cols)
    return lgb.Dataset(X, label=y, params=params, free_raw_data=False)


def train_single_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    cfg: TrainConfig,
) -> Tuple[lgb.Booster, np.ndarray]:
    dtrain = _lgb_dataset(X_tr, y_tr, cfg.categorical_features)
    dvalid = _lgb_dataset(X_va, y_va, cfg.categorical_features)
    params = cfg.params.copy()
    params["seed"] = cfg.seed

    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        early_stopping_rounds=cfg.early_stopping_rounds,
        verbose_eval=cfg.verbose_eval,
    )
    pred_va = model.predict(X_va, num_iteration=model.best_iteration)
    return model, pred_va


def cross_validate_time(
    df: pd.DataFrame,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    cfg: TrainConfig,
) -> CVResult:
    assert cfg.features is not None and len(cfg.features) > 0, "cfg.features 不能为空"

    oof = np.zeros(len(df), dtype=float)
    models: List[lgb.Booster] = []
    fold_scores: List[float] = []
    importances: List[pd.DataFrame] = []

    y = df[cfg.target_col].values

    for i, (tr_idx, va_idx) in enumerate(splits, 1):
        X_tr = df.iloc[tr_idx][cfg.features]
        y_tr = df.iloc[tr_idx][cfg.target_col]
        X_va = df.iloc[va_idx][cfg.features]
        y_va = df.iloc[va_idx][cfg.target_col]

        model, pred_va = train_single_fold(X_tr, y_tr, X_va, y_va, cfg)
        models.append(model)
        oof[va_idx] = pred_va

        score = competition_score(y_va.values, pred_va)
        fold_scores.append(score)
        print(f"[Fold {i}] competition_score = {score:.5f} | best_iter={model.best_iteration}")

        imp_df = pd.DataFrame({
            "feature": cfg.features,
            "importance": model.feature_importance(importance_type="gain"),
            "fold": i,
        })
        importances.append(imp_df)

    fi = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    return CVResult(models=models, oof_pred=oof, fold_scores=fold_scores, feature_importance=fi)


def fit_full(
    df: pd.DataFrame,
    cfg: TrainConfig,
) -> lgb.Booster:
    dtrain = _lgb_dataset(df[cfg.features], df[cfg.target_col], cfg.categorical_features)
    params = cfg.params.copy()
    params["seed"] = cfg.seed
    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=cfg.num_boost_round,
        verbose_eval=cfg.verbose_eval,
    )
    return model


def predict(models: List[lgb.Booster], X: pd.DataFrame) -> np.ndarray:
    preds = np.zeros(len(X), dtype=float)
    for m in models:
        preds += m.predict(X, num_iteration=getattr(m, "best_iteration", None))
    preds /= max(1, len(models))
    return preds


def export_submission(
    test_df: pd.DataFrame,
    preds: np.ndarray,
    out_path: str = "submissions/sub_lgbm_baseline.csv",
    id_col: str = "id",
    value_col: str = "new_house_transaction_amount",
) -> None:
    sub = pd.DataFrame({
        id_col: test_df[id_col].values,
        value_col: preds.astype(float),
    })
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"[export_submission] saved to: {out_path} (rows={len(sub)})")


# =========================
# 启发式基线（无学习器）
# =========================
@dataclass
class HeuristicConfig:
    """
    与 baseline 一致的参数：
    - t1：用于均值的最近月份数
    - t2：零值门控窗口，若过去 t2 个月的最小值为 0，则预测 0
    - method：'geomean' 或 'mean'
    """
    t1: int = 6
    t2: int = 6
    method: str = "geomean"  # 'geomean' or 'mean'


def _last_k_mean(arr: np.ndarray, k: int, method: str = "geomean") -> float:
    """
    计算最近 k 个数的均值：
    - geomean：对非负数做 log1p 再 expm1，避免 0 值造成 -inf
    - mean：普通算术均值
    """
    vec = np.asarray(arr[-k:], dtype=float)
    if method == "geomean":
        # log1p + expm1 以缓解 0 值问题；与 baseline 的纯 log/exp 略有差异，但更稳定
        return float(np.expm1(np.mean(np.log1p(vec))))
    else:
        return float(np.mean(vec))


def baseline_cv_score(
    pivot: pd.DataFrame,
    cfg: HeuristicConfig,
    n_splits: int = 4,
    valid_size_months: int = 12,
) -> Tuple[float, List[float]]:
    """
    基于 time×sector 的透视表做时间 CV 评估，复现 baseline 般的流程。
    - pivot：index 为月份（升序），columns 为 sector_id
    返回：overall_score, per_fold_scores
    """
    times = np.arange(len(pivot))  # 0..T-1
    fold_scores: List[float] = []
    true_list, pred_list = [], []

    # 构造与 baseline 相匹配的切分
    splits = []
    train_end = 19  # 第一个折：0..18 训练
    for _ in range(n_splits):
        valid_start = train_end
        valid_end = valid_start + valid_size_months
        if valid_end > len(times):
            break
        tr_idx = np.arange(0, train_end)
        va_idx = np.arange(valid_start, valid_end)
        splits.append((tr_idx, va_idx))
        train_end = valid_end

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        a_tr = pivot.iloc[tr_idx, :]
        a_va = pivot.iloc[va_idx, :]

        # 对每个验证月都用训练期的最后 t1 个月统计
        pred_rows = {}
        for t in va_idx:
            col_pred = {}
            # 最近 t2 个月是否有 0 -> 则置 0
            zero_mask = (a_tr.tail(cfg.t2).min(axis=0) == 0)
            for sid, series in a_tr.items():
                if zero_mask[sid]:
                    col_pred[sid] = 0.0
                else:
                    col_pred[sid] = _last_k_mean(series.values, cfg.t1, cfg.method)
            pred_rows[t] = col_pred
        a_pred = pd.DataFrame(pred_rows).T
        # 评估
        s = competition_score(a_va.values.ravel(), a_pred.values.ravel())
        fold_scores.append(s)
        true_list.append(a_va.stack())
        pred_list.append(a_pred.stack())
        print(f"[Heuristic Fold {fold}] score={s:.3f}")

    if not fold_scores:
        return 0.0, []
    overall = competition_score(pd.concat(true_list).values, pd.concat(pred_list).values)
    return overall, fold_scores


def baseline_predict_future(
    pivot: pd.DataFrame,
    future_horizons: int,
    cfg: HeuristicConfig,
) -> pd.DataFrame:
    """
    用启发式方法对未来若干个月做预测。
    - 返回 DataFrame：index 为将来“相对时间步”（从 len(pivot) 开始的整数），columns 为 sector_id
    """
    a_tr = pivot.copy()
    pred_rows = {}
    for t in range(future_horizons):
        zero_mask = (a_tr.tail(cfg.t2).min(axis=0) == 0)
        row = {}
        for sid, series in a_tr.items():
            if zero_mask[sid]:
                row[sid] = 0.0
            else:
                row[sid] = _last_k_mean(series.values, cfg.t1, cfg.method)
        # 将该月预测“追加”到历史，以支持递推式的未来多步
        pred_rows[len(a_tr) + t] = row
        a_tr = pd.concat([a_tr, pd.DataFrame([row], index=[a_tr.index.max() + pd.offsets.MonthBegin(1)])])
    pred = pd.DataFrame.from_dict(pred_rows, orient="index")
    pred.index.name = "time"
    return pred
