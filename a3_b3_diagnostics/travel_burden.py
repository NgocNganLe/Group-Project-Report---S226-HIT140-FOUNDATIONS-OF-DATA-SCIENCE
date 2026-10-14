
import io
import os
import re
import numpy as np
import pandas as pd


# Load and clean
def read_fixtures(path):
    
    try:
        text = open(path, encoding="utf-8-sig").read()
    except UnicodeDecodeError:                 
        text = open(path, encoding="cp1252").read()

    df = pd.read_csv(io.StringIO(text))

    if df.shape[1] == 1 and "," in str(df.columns[0]):
        lines = [df.columns[0]] + df.iloc[:, 0].fillna("").astype(str).tolist()
        df = pd.read_csv(io.StringIO("\n".join(lines)))

   
    elif df.shape[1] == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(io.StringIO(text), sep=";")

    return df


fixtures = read_fixtures("data/fixtures.csv")


def repair_text(text):
    """Fix garbled characters such as 'CÃ´te' -> 'Côte' (UTF-8 read as cp1252)."""
    if not isinstance(text, str):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text                            # already clean

needed = ["Date", "Home", "Away", "Venue"]
missing = [c for c in needed if c not in fixtures.columns]
if missing:
    raise ValueError(f"fixtures.csv is missing columns: {missing}. "
                     f"Columns found: {fixtures.columns.tolist()}")

fixtures = fixtures[needed].dropna(subset=["Home", "Away"]).copy()   
fixtures = fixtures[fixtures["Home"] != "Home"]                      


for col in ["Home", "Away", "Venue"]:
    fixtures[col] = fixtures[col].map(repair_text)

fixtures["Home"] = fixtures["Home"].str.strip().str.replace(r"\s+[a-z]{2,3}$", "", regex=True)
fixtures["Away"] = fixtures["Away"].str.strip().str.replace(r"^[a-z]{2,3}\s+", "", regex=True)

dates = fixtures["Date"].astype(str).str.strip()
if dates.str.match(r"^\d{4}-\d{2}-\d{2}").all():             
    fixtures["Date"] = pd.to_datetime(dates, format="ISO8601")
else:                                                             
    fixtures["Date"] = pd.to_datetime(dates, dayfirst=True)
fixtures["Venue"] = (fixtures["Venue"]
                     .str.replace(r"\s+", " ", regex=True)
                     .str.replace(r"\s*\(Neutral Site\)", "", regex=True)
                     .str.strip())

n_teams = pd.concat([fixtures["Home"], fixtures["Away"]]).nunique()
print(f"Matches: {len(fixtures)}  |  distinct teams: {n_teams}  (expect 104 and 48)")
fixtures = fixtures.reset_index(drop=True)
fixtures["match_id"] = fixtures.index + 1        

# load venue co-ordinates
venues = pd.read_csv("data/venues.csv")
venues["venue"] = venues["venue"].str.strip()

unknown = set(fixtures["Venue"]) - set(venues["venue"])
if unknown:
    raise ValueError(f"These venues have no coordinates in venues.csv: {sorted(unknown)}")


side_a = fixtures.assign(team=fixtures["Home"], opponent=fixtures["Away"], side="A")
side_b = fixtures.assign(team=fixtures["Away"], opponent=fixtures["Home"], side="B")
tm = (pd.concat([side_a, side_b])
        .sort_values(["team", "Date", "match_id"])
        .reset_index(drop=True))

tm["prev_venue"] = tm.groupby("team")["Venue"].shift(1)

cur = venues.rename(columns={"venue": "Venue", "lat": "cur_lat", "lon": "cur_lon"})
prev = venues.rename(columns={"venue": "prev_venue", "lat": "prev_lat", "lon": "prev_lon"})
tm = tm.merge(cur, on="Venue", how="left").merge(prev, on="prev_venue", how="left")


if os.path.exists("data/base_camps.csv"):
    camps = pd.read_csv("data/base_camps.csv").rename(
        columns={"lat": "camp_lat", "lon": "camp_lon"})
    tm = tm.merge(camps, on="team", how="left")
    first = tm["prev_venue"].isna()
    tm.loc[first, "prev_lat"] = tm.loc[first, "camp_lat"]
    tm.loc[first, "prev_lon"] = tm.loc[first, "camp_lon"]


#haversine distance
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))


tm["travel_km"] = haversine_km(tm["prev_lat"], tm["prev_lon"],
                               tm["cur_lat"], tm["cur_lon"])

n_missing = tm["travel_km"].isna().sum()
print(f"Team-match rows: {len(tm)}  |  rows with no travel value (opening matches): {n_missing}")

# Team A minus Team B
a = tm[tm["side"] == "A"][["match_id", "Date", "team", "travel_km"]]
a.columns = ["match_id", "Date", "Team_A", "travel_A"]
b = tm[tm["side"] == "B"][["match_id", "team", "travel_km"]]
b.columns = ["match_id", "Team_B", "travel_B"]

out = a.merge(b, on="match_id").sort_values("match_id")
out["Travel_Burden_Diff"] = out["travel_A"] - out["travel_B"]

out.to_csv("data/a3_travel_burden.csv", index=False)
print(out.head(10))
print("\nSaved to data/a3_travel_burden.csv")