
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_dist, shapiro, t as t_dist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "S1_FIFA_Points_Gap",
    "S2_RestDaysGap",
    "S3_Previous_WC_Experience_Share_Gap",
    "S4_FormLast5Gap",
    "B1_Att_Strength_Adj",
    "B2_Opp_Def_Strength_Adj",
    "B3_WC_Rolling_Goals",
    "B4_Host_Nation_Home",
]

TARGET = "goals_scored"
GROUP = "match_id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Model 2.2 OLS, grouped validation and diagnostics."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to the complete 208-row Model 2.2 CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model22_results"),
        help="Folder for CSV and JSON outputs (default: model22_results).",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def find_default_data() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "model22_8_features_208_COMPLETE_HOST_MEDIAN.csv",
        script_dir
        / "outputs"
        / "model22_complete_host_imputed"
        / "model22_8_features_208_COMPLETE_HOST_MEDIAN.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Model 2.2 CSV not found. Supply it with --data, for example:\n"
        "python run_model22.py --data model22_8_features_208_COMPLETE_HOST_MEDIAN.csv"
    )


def validate_data(df: pd.DataFrame) -> None:
    required = [GROUP, "team", "opponent", TARGET, *FEATURES]
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if len(df) != 208:
        raise ValueError(f"Expected 208 team-match rows, found {len(df)}.")
    if df[GROUP].nunique() != 104:
        raise ValueError(f"Expected 104 unique matches, found {df[GROUP].nunique()}.")

    rows_per_match = df.groupby(GROUP).size()
    if not rows_per_match.eq(2).all():
        bad = rows_per_match[~rows_per_match.eq(2)].to_dict()
        raise ValueError(f"Every match must have two rows. Invalid matches: {bad}")

    for match_id, group in df.groupby(GROUP):
        teams = set(group["team"])
        opponents = set(group["opponent"])
        if len(teams) != 2 or teams != opponents:
            raise ValueError(
                f"Invalid row-team/opponent orientation for match_id={match_id}: "
                f"teams={teams}, opponents={opponents}"
            )

    analytic_columns = [TARGET, *FEATURES]
    missing = df[analytic_columns].isna().sum()
    if missing.any():
        raise ValueError(
            "Missing analytic values remain. Resolve them before modelling:\n"
            + missing[missing.gt(0)].to_string()
        )

    numeric = df[analytic_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Predictors and target must contain finite numeric values only.")

    host_values = set(df["B4_Host_Nation_Home"].unique())
    if not host_values.issubset({0, 1}):
        raise ValueError(f"B4_Host_Nation_Home must be binary; found {host_values}.")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def add_constant(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [np.ones(len(frame)), frame.to_numpy(dtype=float)]
    )


def ols_fit(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    fitted = x @ beta
    residuals = y - fitted
    n, k = x.shape
    rss = float(residuals @ residuals)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
    f_value = ((tss - rss) / (k - 1)) / (rss / (n - k))
    return {
        "beta": beta,
        "fitted": fitted,
        "residuals": residuals,
        "xtx_inv": xtx_inv,
        "r2": float(r2),
        "adjusted_r2": float(adjusted_r2),
        "f": float(f_value),
        "f_p": float(f_dist.sf(f_value, k - 1, n - k)),
        "n": n,
        "k": k,
    }


def clustered_covariance(
    fit: dict[str, object], x: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, int]:
    unique_groups = np.unique(groups)
    meat = np.zeros((x.shape[1], x.shape[1]))
    residuals = np.asarray(fit["residuals"])
    for group in unique_groups:
        mask = groups == group
        score = x[mask].T @ residuals[mask]
        meat += np.outer(score, score)

    n = int(fit["n"])
    k = int(fit["k"])
    group_count = len(unique_groups)
    correction = (group_count / (group_count - 1)) * ((n - 1) / (n - k))
    xtx_inv = np.asarray(fit["xtx_inv"])
    covariance = correction * xtx_inv @ meat @ xtx_inv
    return covariance, group_count


def coefficient_table(
    fit: dict[str, object], x: np.ndarray, groups: np.ndarray
) -> tuple[pd.DataFrame, dict[str, float]]:
    covariance, group_count = clustered_covariance(fit, x, groups)
    standard_errors = np.sqrt(np.diag(covariance))
    beta = np.asarray(fit["beta"])
    t_values = beta / standard_errors
    p_values = 2 * t_dist.sf(np.abs(t_values), df=group_count - 1)
    critical_t = t_dist.ppf(0.975, df=group_count - 1)

    slopes = beta[1:]
    slope_covariance = covariance[1:, 1:]
    clustered_f = float(
        slopes @ np.linalg.pinv(slope_covariance) @ slopes / len(slopes)
    )
    clustered_f_p = float(
        f_dist.sf(clustered_f, len(slopes), group_count - 1)
    )

    table = pd.DataFrame(
        {
            "Variable": ["Intercept", *FEATURES],
            "Coefficient": beta,
            "Clustered_SE": standard_errors,
            "t_value": t_values,
            "p_value": p_values,
            "CI_2.5%": beta - critical_t * standard_errors,
            "CI_97.5%": beta + critical_t * standard_errors,
        }
    )
    table["Significant_5pct"] = np.where(table["p_value"] < 0.05, "Yes", "No")
    return table, {
        "clustered_F": clustered_f,
        "clustered_F_p": clustered_f_p,
        "cluster_count": group_count,
    }


def calculate_vif(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    rows = []
    for index, name in enumerate(frame.columns):
        target = values[:, index]
        other_predictors = np.delete(values, index, axis=1)
        x = np.column_stack([np.ones(len(frame)), other_predictors])
        beta = np.linalg.lstsq(x, target, rcond=None)[0]
        predicted = x @ beta
        r2 = 1 - ((target - predicted) ** 2).sum() / (
            (target - target.mean()) ** 2
        ).sum()
        rows.append({"Variable": name, "VIF": float(1 / (1 - r2))})
    return pd.DataFrame(rows)


def breusch_pagan(
    residuals: np.ndarray, x: np.ndarray
) -> dict[str, float]:
    squared_residuals = residuals**2
    auxiliary = ols_fit(x, squared_residuals)
    n = len(squared_residuals)
    degrees_freedom = x.shape[1] - 1
    r2 = float(auxiliary["r2"])
    lm = n * r2
    f_value = (r2 / degrees_freedom) / (
        (1 - r2) / (n - x.shape[1])
    )
    return {
        "LM": float(lm),
        "LM_p": float(chi2.sf(lm, degrees_freedom)),
        "F": float(f_value),
        "F_p": float(
            f_dist.sf(f_value, degrees_freedom, n - x.shape[1])
        ),
    }


def grouped_cross_validation(
    df: pd.DataFrame, folds: int
) -> pd.DataFrame:
    rows = []
    splitter = GroupKFold(n_splits=folds)
    split_iterator = splitter.split(df[FEATURES], df[TARGET], df[GROUP])

    for fold, (train_index, test_index) in enumerate(split_iterator, start=1):
        train = df.iloc[train_index]
        test = df.iloc[test_index]

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
        model.fit(train[FEATURES], train[TARGET])
        predictions = model.predict(test[FEATURES])
        baseline_predictions = np.repeat(train[TARGET].mean(), len(test))

        row = {"fold": fold, **metrics(test[TARGET], predictions)}
        row.update(
            {
                f"baseline_{name}": value
                for name, value in metrics(
                    test[TARGET], baseline_predictions
                ).items()
            }
        )
        row["train_matches"] = int(train[GROUP].nunique())
        row["test_matches"] = int(test[GROUP].nunique())
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    data_path = args.data if args.data is not None else find_default_data()
    data_path = data_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    validate_data(df)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_index, test_index = next(
        splitter.split(df[FEATURES], df[TARGET], groups=df[GROUP])
    )
    train_match_ids = set(df.iloc[train_index][GROUP])
    test_match_ids = set(df.iloc[test_index][GROUP])
    if not train_match_ids.isdisjoint(test_match_ids):
        raise RuntimeError("Match leakage detected between train and test sets.")

    train = df[df[GROUP].isin(train_match_ids)].copy()
    test = df[df[GROUP].isin(test_match_ids)].copy()

    x_train = add_constant(train[FEATURES])
    y_train = train[TARGET].to_numpy(dtype=float)
    fit = ols_fit(x_train, y_train)
    coefficients, clustered_test = coefficient_table(
        fit, x_train, train[GROUP].to_numpy()
    )

    x_test = add_constant(test[FEATURES])
    predictions = x_test @ np.asarray(fit["beta"])
    baseline_predictions = np.repeat(train[TARGET].mean(), len(test))
    holdout_model = metrics(test[TARGET].to_numpy(), predictions)
    holdout_baseline = metrics(
        test[TARGET].to_numpy(), baseline_predictions
    )

    holdout_table = pd.DataFrame(
        [
            {"Model": "Model 2.2", **holdout_model},
            {"Model": "Training-mean baseline", **holdout_baseline},
        ]
    )

    cv = grouped_cross_validation(df, args.folds)
    cv_mean = {
        name: float(cv[name].mean())
        for name in [
            "R2",
            "MAE",
            "RMSE",
            "baseline_R2",
            "baseline_MAE",
            "baseline_RMSE",
        ]
    }
    cv_standard_deviation = {
        name: float(cv[name].std(ddof=1))
        for name in ["R2", "MAE", "RMSE"]
    }

    vif = calculate_vif(train[FEATURES])
    shapiro_statistic, shapiro_p = shapiro(np.asarray(fit["residuals"]))
    bp = breusch_pagan(np.asarray(fit["residuals"]), x_train)

    prediction_table = test[
        [GROUP, "team", "opponent", TARGET]
    ].copy()
    prediction_table["predicted_goals"] = predictions
    prediction_table["baseline_prediction"] = baseline_predictions
    prediction_table["residual"] = (
        prediction_table[TARGET] - prediction_table["predicted_goals"]
    )

    summary = {
        "input_file": str(data_path),
        "data_validation": {
            "rows": int(len(df)),
            "matches": int(df[GROUP].nunique()),
            "predictors": FEATURES,
            "missing_analytic_values": int(
                df[[TARGET, *FEATURES]].isna().sum().sum()
            ),
        },
        "split": {
            "method": "GroupShuffleSplit by match_id",
            "test_size": args.test_size,
            "random_state": args.random_state,
            "train_matches": len(train_match_ids),
            "test_matches": len(test_match_ids),
            "train_rows": len(train),
            "test_rows": len(test),
        },
        "training_ols": {
            "R2": fit["r2"],
            "adjusted_R2": fit["adjusted_r2"],
            "ordinary_F": fit["f"],
            "ordinary_F_p": fit["f_p"],
            **clustered_test,
        },
        "holdout_model": holdout_model,
        "holdout_baseline": holdout_baseline,
        "cross_validation_mean": cv_mean,
        "cross_validation_sd": cv_standard_deviation,
        "training_diagnostics": {
            "scope": "training rows only",
            "max_VIF": float(vif["VIF"].max()),
            "max_VIF_variable": str(
                vif.loc[vif["VIF"].idxmax(), "Variable"]
            ),
            "Shapiro_W": float(shapiro_statistic),
            "Shapiro_p": float(shapiro_p),
            "Breusch_Pagan_LM": bp["LM"],
            "Breusch_Pagan_LM_p": bp["LM_p"],
            "Breusch_Pagan_F": bp["F"],
            "Breusch_Pagan_F_p": bp["F_p"],
        },
    }

    coefficients.to_csv(
        output_dir / "model22_coefficients_clustered.csv",
        index=False,
        encoding="utf-8-sig",
    )
    holdout_table.to_csv(
        output_dir / "model22_holdout_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv.to_csv(
        output_dir / "model22_grouped_cv_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    vif.to_csv(
        output_dir / "model22_training_vif.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_table.to_csv(
        output_dir / "model22_holdout_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "model22_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nMODEL 2.2 COMPLETED")
    print(f"Input: {data_path}")
    print(
        f"Train: {len(train_match_ids)} matches / {len(train)} rows | "
        f"Test: {len(test_match_ids)} matches / {len(test)} rows"
    )
    print(
        f"Training R2 = {fit['r2']:.3f}; "
        f"adjusted R2 = {fit['adjusted_r2']:.3f}"
    )
    print(
        f"Hold-out RMSE: model = {holdout_model['RMSE']:.3f}; "
        f"baseline = {holdout_baseline['RMSE']:.3f}"
    )
    print(
        f"Cross-validated mean RMSE: model = {cv_mean['RMSE']:.3f}; "
        f"baseline = {cv_mean['baseline_RMSE']:.3f}"
    )
    print(
        f"Training diagnostics: max VIF = {vif['VIF'].max():.2f}; "
        f"Breusch-Pagan p = {bp['LM_p']:.3f}"
    )
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
