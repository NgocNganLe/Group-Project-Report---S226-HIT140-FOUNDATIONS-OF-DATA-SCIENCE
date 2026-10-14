
import re
import unicodedata

import numpy as np
import pandas as pd

MATCH_FILE = "data/model22_master.csv"
QUAL_FILE = "data/qualification_match_adjustment_audit.csv"
OUT_FILE = "data/model22_with_B3.csv"
HOSTS = {"canada", "mexico", "usa"}

# Different spellings of the same team in the two files.
SYNONYMS = {
    "south korea": "korea republic", "ivory coast": "cote d ivoire", "turkey": "turkiye",
    "dr congo": "congo dr", "cape verde": "cabo verde", "iran": "ir iran",
    "czech republic": "czechia", "united states": "usa",
    "bosnia and herzegovina": "bosnia", "bosnia herzegovina": "bosnia", "bosnia herz": "bosnia",
}


def team_key(name):
    """Lower-case team name without accents or punctuation, so spellings can be compared."""
    s = str(name).replace("–", " ").replace("—", " ").replace("-", " ")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()
    return SYNONYMS.get(s, s)


def find_column(columns, prefix, exclude=("pre", "elo", "stren", "weight", "adjust", "rel")):
    """First column whose name starts with `prefix`, skipping the Elo and strength columns."""
    for col in columns:
        if col.lower().startswith(prefix) and not any(word in col.lower() for word in exclude):
            return col
    raise ValueError(f"No column starting with '{prefix}'. Columns found: {list(columns)}")


# 1. Load the Model 2.2 dataset
df = pd.read_csv(MATCH_FILE, encoding="utf-8-sig")
required = ["match_id", "match_date", "team", "fixture_side", "goals_scored"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"{MATCH_FILE} is missing {missing}. Columns found: {df.columns.tolist()}")
df["team_key"] = df["team"].map(team_key)
print(f"Rows: {len(df)}  |  teams: {df['team_key'].nunique()} ")

# 2. Mean goals per match in qualification, for each team
qual = pd.read_csv(QUAL_FILE, encoding="utf-8-sig")
team_col = find_column(qual.columns, "team")
opp_col = find_column(qual.columns, "opponent")
goals_for = "goals_for"
goals_against = next(c for c in qual.columns if c.lower().startswith("goals_agai"))
date_col = "date" if "date" in qual.columns else qual.columns[0]

# Each match may be listed from one side only, so build both sides and remove duplicates.
side_1 = qual[[date_col, team_col, opp_col, goals_for]].set_axis(["date", "team", "opp", "goals"], axis=1)
side_2 = qual[[date_col, opp_col, team_col, goals_against]].set_axis(["date", "team", "opp", "goals"], axis=1)
long = pd.concat([side_1, side_2], ignore_index=True)
long["team_key"] = long["team"].map(team_key)
long["opp_key"] = long["opp"].map(team_key)
long = long.drop_duplicates(subset=["date", "team_key", "opp_key"])

world_cup_teams = set(df["team_key"])
qual_mean = long[long["team_key"].isin(world_cup_teams)].groupby("team_key")["goals"].mean()

no_qualifiers = sorted(world_cup_teams - set(qual_mean.index))
unexpected = [t for t in no_qualifiers if t not in HOSTS]
if unexpected:
    raise ValueError(f"No qualification matches found for {unexpected}; check the team spellings.")
print("Teams without qualification matches :", no_qualifiers)

host_value = qual_mean.mean()
print(f"Value used for the hosts (mean of the other teams): {host_value:.3f}")

# 3. Mean goals in earlier World Cup matches (the current match is excluded)
ordered = df.sort_values(["team_key", "match_id"])
goals = ordered.groupby("team_key")["goals_scored"]
n_prior = goals.cumcount()
prior_goals = goals.cumsum() - ordered["goals_scored"]
df["prior_wc_matches"] = n_prior
df["B3_WC_Rolling_Goals"] = prior_goals / n_prior.where(n_prior > 0)

# 4. First match: qualification mean; hosts: mean of the other teams
first = df["prior_wc_matches"] == 0
df["B3_source"] = np.where(first, "qualification", "earlier_wc_matches")
df.loc[first, "B3_WC_Rolling_Goals"] = df.loc[first, "team_key"].map(qual_mean)
is_host = first & df["B3_WC_Rolling_Goals"].isna()
df.loc[is_host, "B3_WC_Rolling_Goals"] = host_value
df.loc[is_host, "B3_source"] = "host_fallback"

# 5. Checks
print("\nSource of the B3 value:")
print(df["B3_source"].value_counts().to_string())
print("Missing B3 values:", int(df["B3_WC_Rolling_Goals"].isna().sum()),)

dates = pd.to_datetime(df["match_date"], format="ISO8601")
not_in_order = df.assign(date=dates).groupby("team_key")["date"].apply(lambda s: not s.is_monotonic_increasing)
print("Teams whose matches are not in date order by match_id:", int(not_in_order.sum()),)

print("\nSpain:")
print(df.loc[df["team_key"] == "spain", ["match_id", "opponent", "goals_scored", "prior_wc_matches",
                                          "B3_WC_Rolling_Goals", "B3_source"]].round(3).to_string(index=False))

# 6. Save
df = df.drop(columns=[c for c in ["team_key", "B3_UNDEFINED_PENDING"] if c in df.columns])
df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
print(f"\nSaved to {OUT_FILE}")
