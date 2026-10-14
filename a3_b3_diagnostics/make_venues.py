
import os
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

IN_FILE = "data/fifa_venue_table.csv"
OUT_FILE = "data/venues.csv"
BACKUP = "data/venues_manual_backup.csv"
COLUMNS = ["venue", "lat", "lon", "query_used", "matched_address"]

geolocator = Nominatim(user_agent="cdu-hit140-wc2026-student-assignment", timeout=15)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.2, max_retries=2, error_wait_seconds=5)

table = pd.read_csv(IN_FILE, encoding="utf-8-sig").fillna("")


done = {}
if os.path.exists(OUT_FILE):
    old = pd.read_csv(OUT_FILE, encoding="utf-8-sig")
    if "matched_address" in old.columns:
        for _, o in old.iterrows():
            if pd.notna(o["lat"]) and pd.notna(o["lon"]):
                done[o["venue"]] = {c: o[c] for c in COLUMNS}
    elif not os.path.exists(BACKUP):
        os.replace(OUT_FILE, BACKUP)
        print(f"Your earlier hand-made venues.csv was kept as {BACKUP}")
print(f"Stadiums in the table: {len(table)}  |  already found: {len(done)}")

if len(done) < len(table):
    print("Checking the connection to OpenStreetMap ...", flush=True)
    try:
        geolocator.geocode("Kansas City, United States")
    except Exception as e:
        raise SystemExit(f"Cannot reach OpenStreetMap ({e}). Check your internet / VPN and run again.")
    print("Connection OK.\n", flush=True)

rows = []
try:
    for i, (_, r) in enumerate(table.iterrows(), start=1):
        if r.venue in done:
            rows.append(done[r.venue])
            continue
        queries = [f"{r.venue}, {r.country}"]
        if r.alt_name:
            queries.append(f"{r.alt_name}, {r.country}")
        found = None
        for q in queries:
            print(f"[{i:2d}/{len(table)}] searching '{q}'", flush=True)
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
            rows.append({"venue": r.venue, "lat": round(hit.latitude, 5), "lon": round(hit.longitude, 5),
                         "query_used": q, "matched_address": hit.address})
            print(f"      -> {hit.latitude:.4f}, {hit.longitude:.4f} | {hit.address[:85]}", flush=True)
        else:
            rows.append({"venue": r.venue, "lat": None, "lon": None, "query_used": "", "matched_address": "NOT FOUND"})
            print("      -> NOT FOUND (look it up on Google Maps and type lat/lon into the csv)", flush=True)
        pd.DataFrame(rows)[COLUMNS].to_csv(OUT_FILE, index=False, encoding="utf-8")
except KeyboardInterrupt:
    pd.DataFrame(rows)[COLUMNS].to_csv(OUT_FILE, index=False, encoding="utf-8")
    raise SystemExit(f"\nStopped. {len(rows)} stadiums saved. Run the script again to continue.")

out = pd.DataFrame(rows)[COLUMNS]
missing = out[out["lat"].isna()]
if len(missing):
    print("\nCHECK - not found:", missing["venue"].tolist())
out.to_csv(OUT_FILE, index=False, encoding="utf-8")
print(f"\nSaved {len(out)} stadiums to {OUT_FILE}")
