"""Regression diagnostics for Model 2.1 and Model 2.2."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

OUT_DIR = "outputs"
SKEW_LIMIT = 1.0     # |skewness| above this: candidate for a transformation
VIF_LIMIT = 5.0
CORR_LIMIT = 0.8
COOK_LIMIT = 1.0     # Cook's distance above this: refit without the row

MODELS = [
    dict(
        name="model_2_1",
        file="data/model21_master.csv",
        target="goal_difference",
        predictors=["fifa_points_diff", "RestDaysGap", "FormLast5Gap", "h2h_goal_diff_hist",
                    "manager_wc_exp_diff", "travel_distance_diff_km",
                    "Previous_WC_Experience_Share_Gap", "Stage_Pressure_Diff"],
        group=None,
        fill={},
    ),
    dict(
        name="model_2_2",
        file="data/model22_with_B3.csv",
        target="goals_scored",
        predictors=["S1_FIFA_Points_Gap", "S2_RestDaysGap", "S3_Previous_WC_Experience_Share_Gap",
                    "S4_FormLast5Gap", "B1_Att_Strength_Adj", "B2_Opp_Def_Strength_Adj",
                    "B3_WC_Rolling_Goals", "B4_Host_Nation_Home"],
        group="match_id",
        # Hosts played no qualifiers: B1 and B2 are filled with the median of the 45 other teams.
        fill={"B1_Att_Strength_Adj": 2.070829, "B2_Opp_Def_Strength_Adj": 0.563194},
    ),
]


def flag(is_bad):
    return "CHECK" if is_bad else "ok"


def load(cfg):
    df = pd.read_csv(cfg["file"], encoding="utf-8-sig")
    columns = cfg["predictors"] + [cfg["target"]] + ([cfg["group"]] if cfg["group"] else [])
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}. Columns in the file: {df.columns.tolist()}")
    data = df[columns].copy()
    data["label"] = df["match_id"] if "match_id" in df.columns else df.index
    for column, value in cfg["fill"].items():
        print(f"Filled {int(data[column].isna().sum())} missing values in {column} with {value}")
        data[column] = data[column].fillna(value)
    n_rows = len(data)
    data = data.dropna().reset_index(drop=True)
    print(f"Rows: {n_rows}  |  dropped for missing values: {n_rows - len(data)}  |  used: {len(data)}")
    return data


def make_splits(groups, n_folds=5, repeats=3, seed=42):
    """Repeated k-fold; rows with the same group value stay in the same fold."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    splits = []
    for _ in range(repeats):
        for fold in np.array_split(rng.permutation(unique), n_folds):
            test = np.isin(groups, fold)
            splits.append((np.where(~test)[0], np.where(test)[0]))
    return splits


def cv_errors(model, X, y, splits):
    rmse, mae = [], []
    for train, test in splits:
        model.fit(X.iloc[train], y.iloc[train])
        error = y.iloc[test].values - model.predict(X.iloc[test])
        rmse.append(np.sqrt(np.mean(error ** 2)))
        mae.append(np.mean(np.abs(error)))
    return float(np.mean(rmse)), float(np.mean(mae))


def mean_only_rmse(y, splits):
    errors = [np.sqrt(np.mean((y.iloc[test].values - y.iloc[train].mean()) ** 2)) for train, test in splits]
    return float(np.mean(errors))


def compare_versions(X, y, groups, skewed, suffix=""):
    """Cross-validated error: raw predictors, Yeo-Johnson on skewed predictors, mean-only baseline."""
    splits = make_splits(groups)
    fit = sm.OLS(y, sm.add_constant(X)).fit()
    rmse, mae = cv_errors(LinearRegression(), X, y, splits)
    rows = [dict(version="Raw predictors" + suffix, cv_RMSE=rmse, cv_MAE=mae,
                 R2=fit.rsquared, adj_R2=fit.rsquared_adj)]
    if skewed:
        transform = ColumnTransformer([("yj", PowerTransformer(method="yeo-johnson"), skewed)],
                                      remainder="passthrough", verbose_feature_names_out=False)
        transform.set_output(transform="pandas")
        rmse_t, mae_t = cv_errors(Pipeline([("yj", transform), ("lr", LinearRegression())]), X, y, splits)
        fit_t = sm.OLS(y, sm.add_constant(transform.fit_transform(X))).fit()
        rows.append(dict(version="Yeo-Johnson on " + ", ".join(skewed) + suffix, cv_RMSE=rmse_t,
                         cv_MAE=mae_t, R2=fit_t.rsquared, adj_R2=fit_t.rsquared_adj))
    rows.append(dict(version="Mean-only baseline" + suffix, cv_RMSE=mean_only_rmse(y, splits),
                     cv_MAE=np.nan, R2=0.0, adj_R2=0.0))
    return pd.DataFrame(rows).round(3)


def plot_heatmap(corr, name):
    labels = list(corr.columns)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title(f"{name}: predictor correlations")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{name}_heatmap.png", dpi=200)
    plt.close(fig)


def plot_residuals(fitted, resid, student, name):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].scatter(fitted, resid, s=14, alpha=0.7)
    axes[0, 0].axhline(0, color="red", lw=1)
    axes[0, 0].set(title="Residuals vs fitted", xlabel="Fitted values", ylabel="Residuals")
    sm.qqplot(student, line="45", ax=axes[0, 1], markersize=3)
    axes[0, 1].set_title("Normal Q-Q plot")
    axes[1, 0].hist(resid, bins=20, edgecolor="white")
    axes[1, 0].set(title="Histogram of residuals", xlabel="Residual")
    axes[1, 1].scatter(fitted, np.sqrt(np.abs(student)), s=14, alpha=0.7)
    axes[1, 1].set(title="Scale-location", xlabel="Fitted values", ylabel="sqrt |standardised residual|")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{name}_residual_plots.png", dpi=200)
    plt.close(fig)


def run(cfg):
    name, target, predictors, group = cfg["name"], cfg["target"], cfg["predictors"], cfg["group"]
    print(f"\n{'=' * 70}\n{name.upper()}   ({cfg['file']})\n{'=' * 70}")
    data = load(cfg)
    X, y, labels = data[predictors], data[target], data["label"].values
    groups = data[group].values if group else np.arange(len(data))
    if len(predictors) != 8:
        print(f"WARNING: expected 8 predictors, found {len(predictors)}.")
    print(f"\nTarget '{target}': mean {y.mean():.2f}, sd {y.std():.2f}, skew {stats.skew(y):.2f}"
          + (f", share of zeros {np.mean(y == 0):.0%}" if group else ""))

    # Correlations and VIF
    corr = X.corr()
    pairs = sorted(((a, b, corr.loc[a, b]) for i, a in enumerate(predictors) for b in predictors[i + 1:]),
                   key=lambda t: -abs(t[2]))
    print("\nStrongest predictor correlations:")
    for a, b, r in pairs[:5]:
        print(f"  {a} vs {b}: r = {r:+.2f}  [{flag(abs(r) > CORR_LIMIT)}]")
    plot_heatmap(corr, name)

    Xc = sm.add_constant(X)
    vif = pd.DataFrame({"predictor": predictors,
                        "VIF": [variance_inflation_factor(Xc.values, i + 1) for i in range(len(predictors))]})
    vif["flag"] = vif["VIF"].map(lambda v: flag(v > VIF_LIMIT))
    vif.round(2).to_csv(f"{OUT_DIR}/{name}_vif.csv", index=False)
    print(f"\nVIF (investigate above {VIF_LIMIT:g}):")
    print(vif.round(2).to_string(index=False))

    # OLS fit and residual plots
    fit = sm.OLS(y, Xc).fit()
    print(f"\nOLS on all rows: R2 = {fit.rsquared:.3f}, adjusted R2 = {fit.rsquared_adj:.3f}, "
          f"F-test p = {fit.f_pvalue:.4f}")
    resid = fit.resid
    influence = fit.get_influence()
    plot_residuals(fit.fittedvalues, resid, influence.resid_studentized_internal, name)

    # Assumption tests
    _, p_shapiro = stats.shapiro(resid)
    _, p_bp, _, _ = het_breuschpagan(resid, Xc)
    p_reset = float(linear_reset(fit, power=2, use_f=True).pvalue)
    cooks = influence.cooks_distance[0]
    top = np.argsort(cooks)[::-1][:3]
    print("\nAssumption tests (p < 0.05 means the assumption is doubtful):")
    print(f"  Normal residuals     Shapiro-Wilk p = {p_shapiro:.4f}  [{flag(p_shapiro < 0.05)}]")
    print(f"  Constant variance    Breusch-Pagan p = {p_bp:.4f}  [{flag(p_bp < 0.05)}]")
    print(f"  Linearity            RESET p = {p_reset:.4f}  [{flag(p_reset < 0.05)}]")
    print(f"  Influential rows     {int(np.sum(cooks > 4 / len(data)))} rows with Cook's distance > 4/n "
          f"(largest {cooks.max():.3f})")
    print("  Largest Cook's distances (match_id, value):",
          [(int(labels[i]), round(float(cooks[i]), 3)) for i in top])

    # Independence of the two rows of a match (Model 2.2)
    if group:
        pair = pd.DataFrame({"g": groups, "r": resid.values})
        pair["k"] = pair.groupby("g").cumcount()
        wide = pair.pivot(index="g", columns="k", values="r")
        print(f"\nIndependence: correlation between the two rows' residuals in the same match = "
              f"{wide.iloc[:, :2].dropna().corr().iloc[0, 1]:+.2f}")
        clustered = sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": groups})
        table = pd.DataFrame({"coef": fit.params, "SE_normal": fit.bse, "SE_clustered": clustered.bse,
                              "p_normal": fit.pvalues, "p_clustered": clustered.pvalues}).round(4)
        table.to_csv(f"{OUT_DIR}/{name}_cluster_robust.csv")
        print("Standard errors treating each match as one cluster:")
        print(table.to_string())
    else:
        robust = sm.OLS(y, Xc).fit(cov_type="HC3")
        table = pd.DataFrame({"coef": fit.params, "SE_normal": fit.bse, "SE_HC3": robust.bse,
                              "p_normal": fit.pvalues, "p_HC3": robust.pvalues}).round(4)
        table.to_csv(f"{OUT_DIR}/{name}_robust.csv")
        print("\nHeteroskedasticity-robust (HC3) standard errors:")
        print(table.to_string())

    # Standardised coefficients
    Xz, yz = (X - X.mean()) / X.std(), (y - y.mean()) / y.std()
    fit_z = sm.OLS(yz, sm.add_constant(Xz)).fit()
    standardised = pd.DataFrame({"std_coef": fit_z.params[predictors], "p_value": fit_z.pvalues[predictors]})
    standardised = standardised.reindex(standardised["std_coef"].abs().sort_values(ascending=False).index).round(4)
    standardised.to_csv(f"{OUT_DIR}/{name}_standardised_coefficients.csv")
    print("\nStandardised coefficients (largest effect first):")
    print(standardised.to_string())
    print(f"Predictors with p < 0.05: {int((fit.pvalues[predictors] < 0.05).sum())} of {len(predictors)}")

    # Transformation: cross-validated error before and after
    skewed = [c for c in predictors if X[c].nunique() > 2 and abs(stats.skew(X[c])) > SKEW_LIMIT]
    print(f"\nSkewed predictors (|skew| > {SKEW_LIMIT:g}): {skewed if skewed else 'none'}")
    comparison = compare_versions(X, y, groups, skewed)
    comparison.to_csv(f"{OUT_DIR}/{name}_before_after.csv", index=False)
    print("\nBefore / after (cross-validation: 5 folds x 3 repeats, matches kept together):")
    print(comparison.to_string(index=False))
    print("Standardising predictors changes the coefficients but not R2 or the errors.")
    tables = [comparison.assign(model=name)]

    # Sensitivity: refit without a very influential row
    worst = int(np.argmax(cooks))
    if cooks[worst] > COOK_LIMIT:
        label = int(labels[worst])
        keep = np.arange(len(data)) != worst
        Xs, ys, gs = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True), groups[keep]
        print(f"\nSensitivity: match {label} has Cook's distance {cooks[worst]:.2f}. Results without that row:")
        skewed_s = [c for c in predictors if Xs[c].nunique() > 2 and abs(stats.skew(Xs[c])) > SKEW_LIMIT]
        sensitivity = compare_versions(Xs, ys, gs, skewed_s, suffix=f" (without match {label})")
        sensitivity.to_csv(f"{OUT_DIR}/{name}_sensitivity.csv", index=False)
        print(sensitivity.to_string(index=False))
        fit_s = sm.OLS(ys, sm.add_constant(Xs)).fit()
        with_without = pd.DataFrame({"coef_all_rows": fit.params[predictors], "p_all_rows": fit.pvalues[predictors],
                                     "coef_without": fit_s.params[predictors],
                                     "p_without": fit_s.pvalues[predictors]}).round(4)
        with_without.to_csv(f"{OUT_DIR}/{name}_coefficients_with_without.csv")
        print("\nCoefficients with and without that row:")
        print(with_without.to_string())
        tables.append(sensitivity.assign(model=name))

    return (fit.fittedvalues.values, resid.values), tables


def report_outputs(residuals, tables):
    fig, axes = plt.subplots(1, len(residuals), figsize=(4.2 * len(residuals), 3.4), squeeze=False)
    for ax, (model, (fitted, resid)) in zip(axes[0], residuals.items()):
        ax.scatter(fitted, resid, s=10, alpha=0.7)
        ax.axhline(0, color="red", lw=1)
        ax.set(title=model.replace("model_", "Model ").replace("_", "."), xlabel="Fitted values",
               ylabel="Residuals")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/report_residual_figure.png", dpi=300)
    plt.close(fig)
    pd.concat(tables)[["model", "version", "cv_RMSE", "cv_MAE", "R2", "adj_R2"]].to_csv(
        f"{OUT_DIR}/report_before_after_table.csv", index=False)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    residuals, tables = {}, []
    for cfg in MODELS:
        if not os.path.exists(cfg["file"]):
            print(f"\nSkipping {cfg['name']}: {cfg['file']} not found.")
            continue
        residuals[cfg["name"]], model_tables = run(cfg)
        tables.extend(model_tables)
    if residuals:
        report_outputs(residuals, tables)
    print(f"\nFigures and tables saved in '{OUT_DIR}'.")
