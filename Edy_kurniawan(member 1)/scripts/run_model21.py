from pathlib import Path
import json
import hashlib
import platform
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FEATURES = ["fifa_points_diff", "RestDaysGap", "FormLast5Gap", "h2h_goal_diff_hist",
            "manager_wc_exp_diff", "travel_distance_diff_km",
            "Previous_WC_Experience_Share_Gap", "Stage_Pressure_Diff"]
TARGET = "goal_difference"


def make_splits(groups, n_folds=5, repeats=3, seed=42):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    splits = []
    for _ in range(repeats):
        for fold in np.array_split(rng.permutation(unique), n_folds):
            test = np.isin(groups, fold)
            splits.append((np.where(~test)[0], np.where(test)[0]))
    return splits


def evaluate(d, model, scenario, version):
    X, y = d[FEATURES], d[TARGET]
    groups = d.match_id.to_numpy()
    records = []
    predictions = []
    for k, (tr, va) in enumerate(make_splits(groups), 1):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        baseline = np.full(len(va), y.iloc[tr].mean())
        error = y.iloc[va].to_numpy() - pred
        base_error = y.iloc[va].to_numpy() - baseline
        records.append({"scenario": scenario, "version": version, "fold": k,
                        "train_n": len(tr), "validation_n": len(va),
                        "RMSE": np.sqrt(np.mean(error**2)), "MAE": np.mean(np.abs(error)),
                        "baseline_RMSE": np.sqrt(np.mean(base_error**2)),
                        "baseline_MAE": np.mean(np.abs(base_error))})
        predictions.extend({"scenario": scenario, "version": version, "fold": k,
                            "match_id": int(mid), "actual": float(yy),
                            "prediction": float(pp), "baseline_prediction": float(bb)}
                           for mid, yy, pp, bb in zip(d.match_id.iloc[va], y.iloc[va], pred, baseline))
    records = pd.DataFrame(records)
    result = {"scenario": scenario, "version": version, "N": len(d),
              "mean_fold_RMSE": records.RMSE.mean(), "SD_fold_RMSE": records.RMSE.std(ddof=1),
              "mean_fold_MAE": records.MAE.mean(),
              "mean_fold_baseline_RMSE": records.baseline_RMSE.mean()}
    return result, records, pd.DataFrame(predictions)


def transformed_model():
    transform = ColumnTransformer(
        [("yeo_johnson", PowerTransformer(method="yeo-johnson"),
          ["RestDaysGap", "Stage_Pressure_Diff"])], remainder="passthrough")
    return Pipeline([("transform", transform), ("OLS", LinearRegression())])


def main():
    OUT.mkdir(exist_ok=True)
    path = ROOT / "data" / "model21_master.csv"
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = ["match_id", "match_date", "team_a", "team_b"] + FEATURES + [TARGET]
    if not set(required).issubset(data):
        raise ValueError("Required Model2.1 columns are missing")
    if len(data) != 104 or not data.match_id.is_unique:
        raise ValueError("Expected104unique match rows")
    if not np.isfinite(data[FEATURES + [TARGET]].to_numpy()).all():
        raise ValueError("Missing/nonfinite model values; no rows silently dropped")
    if not data.loc[data.match_id.eq(13), "RestDaysGap"].eq(-70).all():
        raise ValueError("Match13rest gap differs from audited working snapshot")
    X, y = data[FEATURES], data[TARGET]
    Xc = sm.add_constant(X)
    ols = sm.OLS(y, Xc).fit()
    robust = sm.OLS(y, Xc).fit(cov_type="HC3", use_t=False)
    table = pd.DataFrame({"coefficient": robust.params, "HC3_SE": robust.bse,
                          "HC3_p_z": robust.pvalues,
                          "CI95_low": robust.conf_int()[0], "CI95_high": robust.conf_int()[1]})
    table.index.name = "term"
    table.to_csv(OUT / "model21_full104_coefficients_HC3.csv")
    influence = ols.get_influence()
    pd.DataFrame({"match_id": data.match_id, "Cook_distance": influence.cooks_distance[0],
                  "leverage": influence.hat_matrix_diag}).to_csv(OUT / "model21_influence.csv", index=False)
    pd.DataFrame({"predictor": FEATURES,
                  "VIF": [variance_inflation_factor(Xc.values, i + 1) for i in range(len(FEATURES))]})\
        .to_csv(OUT / "model21_VIF.csv", index=False)
    diagnostics = {"N": 104, "in_sample_R2": ols.rsquared,
                   "in_sample_adj_R2": ols.rsquared_adj,
                   "Shapiro_p": stats.shapiro(ols.resid).pvalue,
                   "Breusch_Pagan_p": het_breuschpagan(ols.resid, Xc)[1],
                   "RESET_p": float(linear_reset(ols, power=2, use_f=True).pvalue)}
    summaries, folds, predictions = [], [], []
    for scenario, d in [("primary_all104", data),
                        ("sensitivity_without13", data[data.match_id.ne(13)].reset_index(drop=True))]:
        for version, model in [("raw", LinearRegression()),
                               ("YeoJohnson_RestDaysGap_StagePressure", transformed_model())]:
            result, record, preds = evaluate(d, model, scenario, version)
            summaries.append(result); folds.append(record); predictions.append(preds)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "model21_CV_summary.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(OUT / "model21_CV_folds.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "model21_CV_predictions.csv", index=False)
    raw = summary[(summary.scenario == "primary_all104") & (summary.version == "raw")].iloc[0]
    reduced = summary[(summary.scenario == "sensitivity_without13") & (summary.version == "raw")].iloc[0]
    checks = {"raw_CV_RMSE_matches_report": abs(raw.mean_fold_RMSE - 2.317) < .0006,
              "raw_baseline_matches_report": abs(raw.mean_fold_baseline_RMSE - 2.000) < .0006,
              "reduced_CV_RMSE_matches_report": abs(reduced.mean_fold_RMSE - 1.671) < .0006,
              "reduced_baseline_matches_report": abs(reduced.mean_fold_baseline_RMSE - 2.033) < .0006,
              "fifa_HC3_p_matches_report": abs(robust.pvalues["fifa_points_diff"] - .0162) < .0001,
              "H2H_HC3_p_matches_report": abs(robust.pvalues["h2h_goal_diff_hist"] - .0370) < .0001}
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise AssertionError(f"Report reproduction failed: {checks}")
    travel = pd.read_csv(ROOT / "data" / "a3_with_master_ids.csv", encoding="utf-8-sig")
    if not travel.match_id.is_unique:
        raise ValueError("Duplicate IDs in travel mapping")
    joined = data.merge(travel[["match_id", "Travel_Burden_Diff"]], on="match_id", how="left", validate="one_to_one")
    if joined.Travel_Burden_Diff.isna().any():
        raise ValueError("Travel mapping incomplete")
    travel_results = []
    for scenario, d in [("all104", joined), ("without13", joined[joined.match_id.ne(13)].reset_index(drop=True))]:
        for definition in ["zero_first_match", "basecamp_first_match"]:
            dd = d.copy()
            if definition == "basecamp_first_match":
                dd["travel_distance_diff_km"] = dd.Travel_Burden_Diff
            result, _, _ = evaluate(dd, LinearRegression(), scenario, definition)
            travel_results.append(result)
    pd.DataFrame(travel_results).to_csv(OUT / "model21_A3_sensitivity.csv", index=False)
    audit = {"status": "REPORT_ALIGNED_WORKING_MODEL21_ONLY_NOT_FINAL_GROUP_DATASET",
             "diagnostics": diagnostics, "report_reproduction_checks": checks,
             "protocol": {"sample": "104matches", "folds": 5, "repeats": 3,
                          "seed": 42, "algorithm": "default_rng permutation of sorted unique match IDs",
                          "aggregation": "mean per-fold RMSE", "HC3_inference": "normal/z"},
             "limits": ["S3roster cutoff is provisional", "Model22notrun",
                        "Collection notebook not rerun", "Removing13changesvalidationpopulation andfolds",
                        "Fixed transformation variables; transform fitted within each trainingfold",
                        "Distance variants have small non-opening coordinate differences"],
             "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [path, ROOT / "data" / "a3_with_master_ids.csv"]},
             "environment": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": scipy.__version__,
                             "statsmodels": statsmodels.__version__, "scikit-learn": sklearn.__version__}}
    (OUT / "REPRODUCTION_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(summary[["scenario", "version", "mean_fold_RMSE", "mean_fold_baseline_RMSE"]].to_string(index=False))


if __name__ == "__main__":
    main()
