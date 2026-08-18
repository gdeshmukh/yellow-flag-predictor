"""Fitting and honest evaluation of the caution model.

The target is a discrete-time hazard: whether a caution begins in the next
``horizon_s``. Two estimators are provided.

``linear`` is ordinary least squares on the 0/1 target -- a linear probability
model. It is the estimator the project was originally aimed at, it is directly
interpretable as "this feature moves caution probability by this much per unit",
and on a rare-event target its coefficients are close to a logistic fit's up to
scale. Its predictions are not bounded to [0, 1] and are clipped before scoring.

``logistic`` is the better-specified choice for a binary outcome and is the
default recommendation once more than a handful of races are available.

Evaluation is grouped by session, always. A random split would put grid points
from the same race -- twenty seconds apart, nearly identical features, sharing a
caution -- on both sides of the split, and would report skill that does not
exist. With ``n`` races the only defensible protocol is leave-one-race-out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features.build import FEATURE_COLUMNS


@dataclass
class FitResult:
    estimator: Pipeline
    columns: list[str]
    coefficients: dict[str, float]
    intercept: float


def _matrix(df: pl.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows at risk, with complete features. Bins already under caution are
    dropped: a caution cannot begin while one is running, so scoring those bins
    would pad the negative class with rows where the answer is structurally no.
    """
    d = df.filter(pl.col("at_risk"))
    d = d.drop_nulls(subset=columns)
    X = d.select(columns).to_numpy().astype(float)
    y = d["y"].to_numpy().astype(int)
    g = d["session_id"].to_numpy()
    return X, y, g


def make_estimator(kind: str = "linear") -> Pipeline:
    if kind == "linear":
        model = LinearRegression()
    elif kind == "logistic":
        # Rare positives: balanced weights stop the fit collapsing to all-zero.
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        raise ValueError(f"unknown estimator {kind!r}; use 'linear' or 'logistic'")
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def _predict(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    model = pipe.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return np.clip(pipe.predict(X), 0.0, 1.0)


def fit(df: pl.DataFrame, *, kind: str = "linear",
        columns: list[str] | None = None) -> FitResult:
    columns = columns or [c for c in FEATURE_COLUMNS if c in df.columns]
    X, y, _ = _matrix(df, columns)
    pipe = make_estimator(kind).fit(X, y)
    model = pipe.named_steps["model"]
    coef = np.ravel(model.coef_)
    return FitResult(
        estimator=pipe,
        columns=columns,
        coefficients=dict(zip(columns, (float(c) for c in coef))),
        intercept=float(np.ravel(model.intercept_)[0]),
    )


def evaluate(df: pl.DataFrame, *, kind: str = "linear",
             columns: list[str] | None = None) -> dict:
    """Leave-one-race-out evaluation against the base rate.

    ``lift_vs_base_rate`` is the number that matters. Predicting the training
    base rate for every bin is a legitimate model; average precision for that
    constant predictor equals the positive rate. Anything at or below 1.0 here
    has learned nothing, whatever its ROC-AUC says.
    """
    columns = columns or [c for c in FEATURE_COLUMNS if c in df.columns]
    X, y, g = _matrix(df, columns)
    n_groups = len(set(g.tolist()))
    if n_groups < 2:
        return {
            "error": "leave-one-race-out needs at least 2 sessions",
            "n_sessions": n_groups,
            "n_rows": int(len(y)),
            "n_positive": int(y.sum()),
            "base_rate": float(y.mean()) if len(y) else None,
        }

    oof = np.full(len(y), np.nan)
    for train, test in LeaveOneGroupOut().split(X, y, groups=g):
        if y[train].sum() == 0:
            continue  # a fold with no positives cannot fit a hazard
        pipe = make_estimator(kind).fit(X[train], y[train])
        oof[test] = _predict(pipe, X[test])

    mask = ~np.isnan(oof)
    yt, pt = y[mask], oof[mask]
    base = float(yt.mean()) if len(yt) else float("nan")
    out = {
        "estimator": kind,
        "n_sessions": n_groups,
        "n_rows": int(len(yt)),
        "n_positive": int(yt.sum()),
        "base_rate": base,
        "n_features": len(columns),
    }
    if yt.sum() == 0 or yt.sum() == len(yt):
        out["error"] = "evaluation fold has a single class"
        return out
    ap = float(average_precision_score(yt, pt))
    out.update({
        "average_precision": ap,
        "base_rate_average_precision": base,
        "lift_vs_base_rate": ap / base if base > 0 else float("nan"),
        "roc_auc": float(roc_auc_score(yt, pt)),
        "brier": float(brier_score_loss(yt, np.clip(pt, 0, 1))),
        "brier_base_rate": float(brier_score_loss(yt, np.full_like(pt, base))),
    })
    return out


def alert_table(y: np.ndarray, p: np.ndarray,
                thresholds: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5)) -> pl.DataFrame:
    """Precision and recall at usable alert rates.

    A caution model is used by raising an alert, so its operating point matters
    more than any summary curve: an alert firing on half the race is useless
    however good its AUC.
    """
    rows = []
    for thr in thresholds:
        pred = p >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        rows.append({
            "threshold": thr,
            "alert_rate": float(pred.mean()),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        })
    return pl.DataFrame(rows)


def collinearity(df: pl.DataFrame, columns: list[str] | None = None,
                 threshold: float = 0.99) -> pl.DataFrame:
    """Feature pairs correlated beyond ``threshold``.

    The legacy dataset shipped ``SPI`` and ``S12`` as separate inputs when
    ``SPI`` was exactly ``(900/11) / S12`` -- the same measurement twice. A pair
    at |r| ~ 1 makes coefficients arbitrary: the fit can move weight between
    them freely, so their individual values mean nothing even though the
    prediction is unaffected. This is a cheap check that the same mistake has
    not reappeared.
    """
    columns = columns or [c for c in FEATURE_COLUMNS if c in df.columns]
    d = df.filter(pl.col("at_risk")).drop_nulls(subset=columns)
    X = d.select(columns).to_numpy().astype(float)
    keep = [i for i in range(X.shape[1]) if np.std(X[:, i]) > 0]
    corr = np.corrcoef(X[:, keep], rowvar=False)
    rows = []
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            r = float(corr[a, b])
            if abs(r) >= threshold:
                rows.append({"feature_a": columns[keep[a]],
                             "feature_b": columns[keep[b]], "r": r})
    return pl.DataFrame(
        rows, schema={"feature_a": pl.Utf8, "feature_b": pl.Utf8, "r": pl.Float64}
    )
