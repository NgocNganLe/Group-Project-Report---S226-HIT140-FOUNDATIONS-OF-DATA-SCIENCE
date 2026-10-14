Member 3: A3 (travel distance), B3 (rolling goals) and regression diagnostics


Scripts
  make_venues.py          looks up stadium coordinates on OpenStreetMap -> data/venues.csv
  make_base_camps.py      looks up FIFA base-camp coordinates -> data/base_camps.csv
  travel_burden.py        A3: km travelled to each match venue, from the previous venue
                          (first match: base camp) -> data/a3_travel_burden.csv
  a3_map_to_master.py     links A3 to the Model 2.1 match_id -> data/a3_with_master_ids.csv
  a3_sensitivity.py       Model 2.1 with first match = 0 and first match = base camp
  b3_wc_rolling_goals.py  B3: mean goals in earlier World Cup matches -> data/model22_with_B3.csv
  diagnostics.py          assumption checks, VIF, transformations, sensitivity -> outputs/

Run order
  1 make_venues.py   2 make_base_camps.py (both need internet)   3 travel_burden.py
  4 a3_map_to_master.py   5 a3_sensitivity.py   6 b3_wc_rolling_goals.py   7 diagnostics.py

Packages: pandas, numpy, statsmodels, scipy, matplotlib, scikit-learn, geopy

Notes
  [A3 in the models uses first match = 0. a3_sensitivity.py shows that using the base camp
   instead changes cross-validated RMSE by under 0.01.]
  
