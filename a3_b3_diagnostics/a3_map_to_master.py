
import re
import unicodedata

import pandas as pd

DATASET = "data/model21_master.csv"
TRAVEL = "data/a3_travel_burden.csv"
OUT = "data/a3_with_master_ids.csv"

SYNONYMS = {"south korea": "korea republic", "ivory coast": "cote d ivoire", "turkey": "turkiye",
            "dr congo": "congo dr", "cape verde": "cabo verde", "iran": "ir iran",
            "czech republic": "czechia", "united states": "usa",
            "bosnia and herzegovina": "bosnia", "bosnia herzegovina": "bosnia", "bosnia herz": "bosnia"}


def team_key(name):
    """Lower-case team name without accents or punctuation, so spellings can be compared."""
    s = str(name).replace("–", " ").replace("—", " ").replace("-", " ")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()
    return SYNONYMS.get(s, s)


dataset = pd.read_csv(DATASET, encoding="utf-8-sig")
travel = pd.read_csv(TRAVEL, encoding="utf-8-sig")
for name, df, cols in [(DATASET, dataset, ["match_id", "match_date", "team_a", "team_b", "travel_distance_diff_km"]),
                       (TRAVEL, travel, ["match_id", "Date", "Team_A", "Team_B", "travel_A", "travel_B",
                                         "Travel_Burden_Diff"])]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing {missing}. Columns found: {df.columns.tolist()}")

dataset["a"] = dataset["team_a"].map(team_key)
dataset["b"] = dataset["team_b"].map(team_key)
travel["a"] = travel["Team_A"].map(team_key)
travel["b"] = travel["Team_B"].map(team_key)
travel = travel.rename(columns={"match_id": "schedule_order_id"})
travel["swapped"] = False

# Copy with Team A and Team B swapped (sign flipped), used if the order differs between the files.
flipped = travel.copy()
flipped["a"], flipped["b"] = travel["b"], travel["a"]
flipped["travel_A"], flipped["travel_B"] = travel["travel_B"], travel["travel_A"]
flipped["Travel_Burden_Diff"] = -travel["Travel_Burden_Diff"]
flipped["swapped"] = True
lookup = pd.concat([travel, flipped], ignore_index=True)

merged = dataset.merge(lookup[["a", "b", "schedule_order_id", "Date", "travel_A", "travel_B",
                               "Travel_Burden_Diff", "swapped"]], on=["a", "b"], how="left")

not_found = merged[merged["schedule_order_id"].isna()]
if len(not_found):
    raise ValueError("No travel row found for these matches (check the team spellings):\n"
                     + not_found[["match_id", "team_a", "team_b"]].to_string(index=False))
if merged["match_id"].duplicated().any():
    raise ValueError("Some matches were linked to more than one row; add the date to the link.")

print(f"Matched {len(merged)} of {len(dataset)} matches  |  Team A / Team B order flipped for {int(merged['swapped'].sum())}")
different_date = merged[pd.to_datetime(merged["match_date"]).dt.date != pd.to_datetime(merged["Date"]).dt.date]
print(f"Matches with a different date in the two files: {len(different_date)} (expected 0)")

merged["difference_km"] = merged["Travel_Burden_Diff"] - merged["travel_distance_diff_km"]
# match_id 1-24 are the opening matches, where the two definitions differ by construction
later = merged[merged["match_id"] > 24]
print(f"\nFrom matchday 2 on ({len(later)} matches), difference between the two travel columns:")
print(f"  largest {later['difference_km'].abs().max():.3f} km  |  mean {later['difference_km'].abs().mean():.3f} km")
print("  matches differing by more than 1 km:", later.loc[later["difference_km"].abs() > 1, "match_id"].tolist())

out = merged[["match_id", "match_date", "team_a", "team_b", "Travel_Burden_Diff", "travel_distance_diff_km",
              "difference_km", "schedule_order_id"]].sort_values("match_id")
out.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\nSaved to {OUT}")