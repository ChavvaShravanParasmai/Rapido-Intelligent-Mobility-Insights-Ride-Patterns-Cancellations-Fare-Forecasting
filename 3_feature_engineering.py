"""
STEP 3 - FEATURE ENGINEERING
-----------------------------
Here we combine (merge) the bookings data with customer and driver
data, and create the extra features suggested in the project brief:

- Fare_per_KM        (already made in cleaning step)
- Fare_per_Min       (already made in cleaning step)
- Rush_Hour_Flag     (already made in cleaning step)
- Long_Distance_Flag (already made in cleaning step)
- City_Pair          (already made in cleaning step)
- Driver_Reliability_Score
- Customer_Loyalty_Score

The final merged & feature-rich table is saved as
'data/cleaned/master_dataset.csv' and is what we use for modeling.
"""

import pandas as pd
import os

CLEAN_DIR = "data/cleaned"

bookings = pd.read_csv(f"{CLEAN_DIR}/bookings_clean.csv")
customers = pd.read_csv(f"{CLEAN_DIR}/customers_clean.csv")
drivers = pd.read_csv(f"{CLEAN_DIR}/drivers_clean.csv")

# ---------------------------------------------------------
# 1. Driver Reliability Score
# ---------------------------------------------------------
# Simple formula: high acceptance rate + low delay rate + good rating = reliable
# We scale everything to 0-1 range and average it.
drivers["driver_reliability_score"] = (
    drivers["acceptance_rate"] * 0.4
    + (1 - drivers["delay_rate"]) * 0.4
    + (drivers["avg_driver_rating"] / 5) * 0.2
).round(3)

# ---------------------------------------------------------
# 2. Customer Loyalty Score
# ---------------------------------------------------------
# Simple formula: more completed rides + low cancellation rate + good rating
# = more loyal customer.
customers["completion_rate"] = (
    customers["completed_rides"] / customers["total_bookings"].replace(0, 1)
)
customers["customer_loyalty_score"] = (
    customers["completion_rate"] * 0.5
    + (1 - customers["cancellation_rate"]) * 0.3
    + (customers["avg_customer_rating"] / 5) * 0.2
).round(3)

# ---------------------------------------------------------
# 3. Merge bookings with customer & driver info
# ---------------------------------------------------------
master = bookings.merge(customers, on="customer_id", how="left")
master = master.merge(drivers, on="driver_id", how="left", suffixes=("", "_driver"))

print("Master dataset shape:", master.shape)
print("Missing values after merge:\n", master.isna().sum().sum())

# A few rows may not match a customer/driver (edge case) - fill with 0/Unknown
master = master.fillna(0)

# ---------------------------------------------------------
# 4. Save master dataset
# ---------------------------------------------------------
master.to_csv(f"{CLEAN_DIR}/master_dataset.csv", index=False)
print("\nFeature engineering complete. Saved: data/cleaned/master_dataset.csv")
