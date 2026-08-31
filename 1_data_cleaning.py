"""
STEP 1 - DATA CLEANING
----------------------
This script loads the 5 raw CSV files, cleans them, and saves a
clean version into the 'data/cleaned/' folder so the next steps
(EDA, feature engineering, modeling) can use tidy data.

What we do here:
1. Load all 5 csv files
2. Fix missing values
3. Convert date/time columns to proper datetime type
4. Create a few simple new columns (features)
5. Save cleaned files
"""

import pandas as pd
import os

# ---------------------------------------------------------
# 1. Load the raw data
# ---------------------------------------------------------
DATA_DIR = "data"
CLEAN_DIR = "data/cleaned"
os.makedirs(CLEAN_DIR, exist_ok=True)

bookings = pd.read_csv(f"{DATA_DIR}/bookings.csv")
customers = pd.read_csv(f"{DATA_DIR}/customers.csv")
drivers = pd.read_csv(f"{DATA_DIR}/drivers.csv")
location_demand = pd.read_csv(f"{DATA_DIR}/location_demand.csv")
time_features = pd.read_csv(f"{DATA_DIR}/time_features.csv")

print("Raw shapes:")
print("bookings:", bookings.shape)
print("customers:", customers.shape)
print("drivers:", drivers.shape)
print("location_demand:", location_demand.shape)
print("time_features:", time_features.shape)

# ---------------------------------------------------------
# 2. Fix missing values in bookings.csv
# ---------------------------------------------------------
# actual_ride_time_min is empty when a ride was Cancelled or Incomplete.
# That is expected (there was no actual ride), so we fill it with 0
# instead of dropping the row.
bookings["actual_ride_time_min"] = bookings["actual_ride_time_min"].fillna(0)

# incomplete_ride_reason is empty when the ride was Completed
# (no reason needed). We fill it with the text "Not Applicable".
bookings["incomplete_ride_reason"] = bookings["incomplete_ride_reason"].fillna(
    "Not Applicable"
)

# ---------------------------------------------------------
# 3. Convert date & time columns to proper datetime type
# ---------------------------------------------------------
# Combine booking_date + booking_time into one datetime column
bookings["booking_datetime"] = pd.to_datetime(
    bookings["booking_date"] + " " + bookings["booking_time"]
)

# time_features.csv already has a datetime column
time_features["datetime"] = pd.to_datetime(time_features["datetime"])

# ---------------------------------------------------------
# 4. Create a few simple new columns
# ---------------------------------------------------------
# Rush hour flag: rides between 8-10 AM or 5-8 PM are "rush hour"
bookings["rush_hour_flag"] = bookings["hour_of_day"].apply(
    lambda h: 1 if (8 <= h <= 10) or (17 <= h <= 20) else 0
)

# Long distance flag: rides longer than 15 km are "long distance"
bookings["long_distance_flag"] = bookings["ride_distance_km"].apply(
    lambda d: 1 if d > 15 else 0
)

# Fare per km and fare per min (avoid divide-by-zero using replace)
bookings["fare_per_km"] = bookings["booking_value"] / bookings["ride_distance_km"].replace(0, 0.1)
bookings["fare_per_min"] = bookings["booking_value"] / bookings["estimated_ride_time_min"].replace(0, 0.1)

# City pair column (pickup + drop) - useful for popular-route analysis
bookings["city_pair"] = bookings["pickup_location"] + "_to_" + bookings["drop_location"]

# ---------------------------------------------------------
# 5. Basic sanity checks
# ---------------------------------------------------------
print("\nMissing values left in bookings after cleaning:")
print(bookings.isna().sum().sum())

# ---------------------------------------------------------
# 6. Save cleaned files
# ---------------------------------------------------------
bookings.to_csv(f"{CLEAN_DIR}/bookings_clean.csv", index=False)
customers.to_csv(f"{CLEAN_DIR}/customers_clean.csv", index=False)
drivers.to_csv(f"{CLEAN_DIR}/drivers_clean.csv", index=False)
location_demand.to_csv(f"{CLEAN_DIR}/location_demand_clean.csv", index=False)
time_features.to_csv(f"{CLEAN_DIR}/time_features_clean.csv", index=False)

print("\nCleaning complete. Clean files saved in data/cleaned/")
