"""
STEP 5 - DATA MANAGEMENT USING SQL
------------------------------------
This script loads our cleaned CSV files into a simple SQLite
database (SQLite needs no server/installation - a single file
database, perfect for beginners).

Tables created:
- bookings
- customers
- drivers
- location_demand
- time_features

We also run a few example SQL queries to show the database works,
and to answer some of the business questions from the brief.
"""

import pandas as pd
import sqlite3
import os

DB_PATH = "rapido.db"
CLEAN_DIR = "data/cleaned"

# Remove old database file if it exists, so we always start fresh
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

# ---------------------------------------------------------
# 1. Connect to (create) the SQLite database
# ---------------------------------------------------------
conn = sqlite3.connect(DB_PATH)

# ---------------------------------------------------------
# 2. Load each cleaned CSV and write it as a SQL table
# ---------------------------------------------------------
bookings = pd.read_csv(f"{CLEAN_DIR}/bookings_clean.csv")
customers = pd.read_csv(f"{CLEAN_DIR}/customers_clean.csv")
drivers = pd.read_csv(f"{CLEAN_DIR}/drivers_clean.csv")
location_demand = pd.read_csv(f"{CLEAN_DIR}/location_demand_clean.csv")
time_features = pd.read_csv(f"{CLEAN_DIR}/time_features_clean.csv")

bookings.to_sql("bookings", conn, if_exists="replace", index=False)
customers.to_sql("customers", conn, if_exists="replace", index=False)
drivers.to_sql("drivers", conn, if_exists="replace", index=False)
location_demand.to_sql("location_demand", conn, if_exists="replace", index=False)
time_features.to_sql("time_features", conn, if_exists="replace", index=False)

# ---------------------------------------------------------
# 3. Add indexes for faster queries (SQL best practice)
# ---------------------------------------------------------
cursor = conn.cursor()
cursor.execute("CREATE INDEX IF NOT EXISTS idx_booking_status ON bookings(booking_status);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_booking_city ON bookings(city);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_id ON bookings(customer_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_driver_id ON bookings(driver_id);")
conn.commit()

print(f"Database created at: {DB_PATH}")
print("Tables: bookings, customers, drivers, location_demand, time_features")

# ---------------------------------------------------------
# 4. Example SQL queries (answers to business questions)
# ---------------------------------------------------------

print("\n--- Query 1: Cancellation rate by city ---")
q1 = """
SELECT city,
       COUNT(*) AS total_rides,
       SUM(CASE WHEN booking_status = 'Cancelled' THEN 1 ELSE 0 END) AS cancelled_rides,
       ROUND(100.0 * SUM(CASE WHEN booking_status = 'Cancelled' THEN 1 ELSE 0 END) / COUNT(*), 2) AS cancellation_rate_pct
FROM bookings
GROUP BY city
ORDER BY cancellation_rate_pct DESC;
"""
print(pd.read_sql(q1, conn))

print("\n--- Query 2: Peak cancellation hours ---")
q2 = """
SELECT hour_of_day,
       COUNT(*) AS total_rides,
       SUM(CASE WHEN booking_status = 'Cancelled' THEN 1 ELSE 0 END) AS cancelled_rides
FROM bookings
GROUP BY hour_of_day
ORDER BY cancelled_rides DESC
LIMIT 5;
"""
print(pd.read_sql(q2, conn))

print("\n--- Query 3: Average fare by vehicle type ---")
q3 = """
SELECT vehicle_type,
       ROUND(AVG(booking_value), 2) AS avg_fare,
       ROUND(AVG(ride_distance_km), 2) AS avg_distance_km
FROM bookings
GROUP BY vehicle_type;
"""
print(pd.read_sql(q3, conn))

print("\n--- Query 4: Top 5 most reliable drivers ---")
q4 = """
SELECT driver_id, driver_city, avg_driver_rating, acceptance_rate, delay_rate
FROM drivers
ORDER BY avg_driver_rating DESC, delay_rate ASC
LIMIT 5;
"""
print(pd.read_sql(q4, conn))

print("\n--- Query 5: Customers with highest cancellation rate ---")
q5 = """
SELECT customer_id, customer_city, total_bookings, cancellation_rate
FROM customers
WHERE total_bookings >= 5
ORDER BY cancellation_rate DESC
LIMIT 5;
"""
print(pd.read_sql(q5, conn))

conn.close()
print("\nSQL database setup and sample queries complete.")
