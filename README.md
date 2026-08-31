# Rapido: Intelligent Mobility Insights
### Ride Patterns, Cancellations & Fare Forecasting

A beginner-friendly, end-to-end Machine Learning project on ride-hailing data.
It cleans the raw data, explores it, engineers features, stores it in a SQL
database, trains two ML models, and shows everything in an interactive
Streamlit dashboard.

---

## 📁 Project Structure

```
rapido_project/
│
├── data/
│   ├── bookings.csv               (raw data)
│   ├── customers.csv              (raw data)
│   ├── drivers.csv                (raw data)
│   ├── location_demand.csv        (raw data)
│   ├── time_features.csv          (raw data)
│   └── cleaned/                   (created after running step 1-3)
│       ├── bookings_clean.csv
│       ├── customers_clean.csv
│       ├── drivers_clean.csv
│       ├── location_demand_clean.csv
│       ├── time_features_clean.csv
│       └── master_dataset.csv     (final merged data used for ML)
│
├── outputs/                       (EDA charts, created after step 2)
├── models/                        (trained ML models, created after step 4)
├── rapido.db                      (SQLite database, created after step 5)
│
├── 1_data_cleaning.py             STEP 1 - clean the raw data
├── 2_eda.py                       STEP 2 - exploratory data analysis (charts)
├── 3_feature_engineering.py       STEP 3 - merge tables + new features
├── 4_model_training.py            STEP 4 - train & evaluate ML models
├── 5_sql_database.py              STEP 5 - build SQLite database + sample queries
├── app.py                         STEP 6 - Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run This Project (in order)

### 1. Install the required libraries
```bash
pip install -r requirements.txt
```

### 2. Run each script in order
```bash
python 1_data_cleaning.py
python 2_eda.py
python 3_feature_engineering.py
python 4_model_training.py
python 5_sql_database.py
```

Each script prints its progress in the terminal and saves its output
(cleaned data, charts, models, or a database file) into the matching folder.

### 3. Launch the dashboard
```bash
streamlit run app.py
```
This opens a browser window with 4 pages (choose from the sidebar):
- **Dashboard** — filters, KPIs, and charts (ride volume, cancellation
  rates by city/hour, fare by vehicle type, etc.)
- **Predict a Ride** — enter ride details (distance, city, traffic,
  weather, etc.) and get an instant prediction of the ride outcome
  (Completed/Cancelled/Incomplete) and the expected fare.
- **Customer Cancellation Risk** — enter a customer's profile and get
  their probability of cancelling a booking.
- **Driver Delay Risk** — enter a driver's profile and get their
  probability of causing a delay.

---

## 🎯 Project Objective

Rapido collects large amounts of trip-level data but wasn't using it to its
full potential. This project turns that raw data into a decision-support
system that can:
- Predict whether a ride will complete, get cancelled, or end incomplete
- Estimate the fare of a ride before it starts
- Identify high cancellation-risk customers and unreliable drivers
- Show operational insights (peak cancellation hours, city hotspots, etc.)

## 🧠 Models Built

| Model | Type | Target | Key Result |
|---|---|---|---|
| Ride Outcome Prediction | Multi-class Classification | booking_status (Completed/Cancelled/Incomplete) | ~73% accuracy |
| Fare Prediction | Regression | booking_value | RMSE ≈ 3.6% of average fare (target was within 10%) |
| Customer Cancellation Risk | Binary Classification | customer_cancel_flag | ~60% accuracy, AUC ≈ 0.63 |
| Driver Delay Prediction | Binary Classification | driver_delay_flag | ~72% accuracy, AUC ≈ 0.80 |

*Note: Ride outcome accuracy came out around 73%, a bit below the 85-90%
benchmark in the brief. This is common with synthetic/simulated data where
some randomness is baked in — the model still finds real, useful patterns
(customer loyalty score, cancellation history, and surge multiplier were the
strongest signals). You can try tuning `n_estimators`/`max_depth` in
`4_model_training.py`, or adding more features, to try to improve this.*

*Important note on Models 3 & 4: the raw dataset's `customer_cancel_flag`
and `driver_delay_flag` columns turned out to be a simple rule based
directly on `cancellation_rate` (>0.2) and `delay_rate` (>0.1). Using those
rate columns as inputs would have given a "perfect" 100% accuracy model —
but that's not real learning, it's the model looking up the answer
(data leakage). So these two models deliberately leave those columns out
and only use genuinely independent signals (rating, tenure, acceptance
rate, experience, traffic, time of day). Their accuracy is lower as a
result, but the predictions are honest and would generalize to new data.*

## 🗃️ SQL Database

`5_sql_database.py` loads all cleaned tables into a single-file SQLite
database (`rapido.db`) — no server setup needed. It includes indexes on
commonly filtered columns and runs 5 example business queries (cancellation
rate by city, peak cancellation hours, average fare by vehicle type, most
reliable drivers, and customers with the highest cancellation rates).

You can open `rapido.db` with any SQLite viewer (e.g. "DB Browser for
SQLite") or query it yourself:
```python
import sqlite3, pandas as pd
conn = sqlite3.connect("rapido.db")
pd.read_sql("SELECT * FROM bookings LIMIT 5;", conn)
```

## 📊 New Features Created

| Feature | Meaning |
|---|---|
| `fare_per_km` | booking_value ÷ ride_distance_km |
| `fare_per_min` | booking_value ÷ estimated_ride_time_min |
| `rush_hour_flag` | 1 if hour is 8-10 AM or 5-8 PM |
| `long_distance_flag` | 1 if ride is longer than 15 km |
| `city_pair` | pickup location + drop location combined |
| `driver_reliability_score` | weighted score from acceptance rate, delay rate, rating |
| `customer_loyalty_score` | weighted score from completion rate, cancellation rate, rating |

## 🛠️ Skills Used
Python scripting, Data Cleaning, Exploratory Data Analysis, Feature
Engineering, Machine Learning (Random Forest), SQL (SQLite), and Streamlit
for the interactive dashboard.
