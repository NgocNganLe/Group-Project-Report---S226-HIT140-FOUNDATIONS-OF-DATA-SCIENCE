"""Model 2.1 with two definitions of travel distance: first match = 0, and first match = FIFA base camp."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from diagnostics import MODELS, cv_errors, make_splits

cfg = MODELS[0]
predictors, target = cfg["predictors"], cfg["target"]
travel_col = "travel_distance_diff_km"

dataset = pd.read_csv(cfg["file"], encoding="utf-8-sig")
base_camp = pd.read_csv("data/a3_with_master_ids.csv", encoding="utf-8-sig")[["match_id", "Travel_Burden_Diff"]]
data = dataset.merge(base_camp, on="match_id", how="left")
if data["Travel_Burden_Diff"].isna().any():
    raise ValueError("Some matches have no base-camp value; run a3_map_to_master.py first.")

differs = (data["Travel_Burden_Diff"] - data[travel_col]).abs() > 1
print(f"Matches where the two definitions differ by more than 1 km: {int(differs.sum())} of {len(data)}")
print(f"Correlation between the two definitions: {data[travel_col].corr(data['Travel_Burden_Diff']):.3f}\n")


def evaluate(label, rows, travel):
    d = data[rows].reset_index(drop=True)
    X = d[predictors].copy()
    X[travel_col] = travel[rows].reset_index(drop=True)
    y = d[target]
    fit = sm.OLS(y, sm.add_constant(X)).fit()
    rmse, mae = cv_errors(LinearRegression(), X, y, make_splits(np.arange(len(d))))
    return dict(travel_definition=label, R2=fit.rsquared, adj_R2=fit.rsquared_adj,
                travel_coef=fit.params[travel_col], travel_p=fit.pvalues[travel_col],
                cv_RMSE=rmse, cv_MAE=mae)


all_rows = pd.Series(True, index=data.index)
without_13 = data["match_id"] != 13
results = pd.DataFrame([
    evaluate("First match = 0, all matches", all_rows, data[travel_col]),
    evaluate("First match = FIFA base camp, all matches", all_rows, data["Travel_Burden_Diff"]),
    evaluate("First match = 0, without match 13", without_13, data[travel_col]),
    evaluate("First match = FIFA base camp, without match 13", without_13, data["Travel_Burden_Diff"]),
]).round(4)
print(results.to_string(index=False))
results.to_csv("outputs/a3_travel_definition_comparison.csv", index=False)