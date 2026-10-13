
import os
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

IN_FILE = "data/fifa_base_camp_table.csv"
OUT_FILE = "data/base_camps.csv"
COLUMNS = ["team", "fifa_city", "training_site", "lat", "lon", "query_used", "matched_address"]

geolocator = Nominatim(user_agent="cdu-hit140-wc2026-student-assignment", timeout=15)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.2, max_retries=2, error_wait_seconds=5)

table = pd.read_csv(IN_FILE, encoding="utf-8-sig")


done = {}
if os.path.exists(OUT_FILE):
    old = pd.read_csv(OUT_FILE, encoding="utf-8-sig")
    if set(COLUMNS).issubset(old.columns):
        for _, o in old.iterrows():
            if pd.notna(o["lat"]) and pd.notna(o["lon"]):
                done[o["team"]] = {c: o[c] for c in COLUMNS}
print(f"Teams in FIFA's table: {len(table)}  |  already found in an earlier run: {len(done)}")


if len(done) < len(table):
    print("Checking the connection to OpenStreetMap ...", flush=True)
    try:
        test = geolocator.geocode("Kansas City, United States")
    except Exception as e:
        raise SystemExit(f"Cannot reach OpenStreetMap ({e}). Check your internet / VPN / university network "
                         f"and run again.")
    print("Connection OK. Starting the lookups (about 3-5 minutes) ...\n", flush=True)

rows = []
try:
    for i, (_, r) in enumerate(table.iterrows(), start=1):
        if r.team in done:
            rows.append(done[r.team])
            continue

      
        queries = [
            f"{r.training_site}, {r.fifa_city}, {r.country}",
            f"{r.training_site}, {r.country}",
            f"{r.fifa_city}, {r.country}",
        ]
        found = None
        for q in queries:
            print(f"[{i:2d}/{len(table)}] {r.team}: searching '{q}'", flush=True)
            try:
                hit = geocode(q)
            except Exception as e:
                print(f"      (lookup error: {e})", flush=True)
                hit = None
            if hit is not None:
                found = (q, hit)
                break

        if found:
            q, hit = found
            rows.append({"team": r.team, "fifa_city": r.fifa_city, "training_site": r.training_site,
                         "lat": round(hit.latitude, 5), "lon": round(hit.longitude, 5),
                         "query_used": q, "matched_address": hit.address})
            print(f"      -> {hit.latitude:.4f}, {hit.longitude:.4f} | {hit.address[:85]}", flush=True)
        else:
            rows.append({"team": r.team, "fifa_city": r.fifa_city, "training_site": r.training_site,
                         "lat": None, "lon": None, "query_used": "", "matched_address": "NOT FOUND"})
            print("      -> NOT FOUND (look it up on Google Maps and type lat/lon into the csv)", flush=True)

        
        pd.DataFrame(rows)[COLUMNS].to_csv(OUT_FILE, index=False, encoding="utf-8")
except KeyboardInterrupt:
    pd.DataFrame(rows)[COLUMNS].to_csv(OUT_FILE, index=False, encoding="utf-8")
    raise SystemExit(f"\nStopped. {len(rows)} teams saved. Run the script again to continue.")

out = pd.DataFrame(rows)[COLUMNS]


north_america = out["lat"].between(14, 60) & out["lon"].between(-170, -50)
bad = out[(~north_america) & out["lat"].notna()]
if len(bad):
    print("\nCHECK - these points are outside North America:", bad["team"].tolist())
missing = out[out["lat"].isna()]
if len(missing):
    print("\nCHECK - not found:", missing["team"].tolist())

out.to_csv(OUT_FILE, index=False, encoding="utf-8")
print(f"\nSaved {len(out)} teams to {OUT_FILE}")
print("Read the 'matched_address' column: each address should be in the right city/state/country.")
print("Source to cite: OpenStreetMap contributors, via the Nominatim geocoding service.")