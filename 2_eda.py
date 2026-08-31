"""
STEP 2 - EXPLORATORY DATA ANALYSIS (EDA)
-----------------------------------------
This script creates simple charts to understand the data better.
All charts are saved as .png images inside the 'outputs/' folder,
so you can add them to your report/README or presentation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_style("whitegrid")

# ---------------------------------------------------------
# Load cleaned data
# ---------------------------------------------------------
bookings = pd.read_csv("data/cleaned/bookings_clean.csv")
drivers = pd.read_csv("data/cleaned/drivers_clean.csv")
customers = pd.read_csv("data/cleaned/customers_clean.csv")

# ---------------------------------------------------------
# 1. Ride volume by hour of day
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
bookings["hour_of_day"].value_counts().sort_index().plot(kind="bar", color="steelblue")
plt.title("Ride Volume by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Rides")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_ride_volume_by_hour.png")
plt.close()

# ---------------------------------------------------------
# 2. Ride volume by weekday
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
bookings["day_of_week"].value_counts().reindex(order).plot(kind="bar", color="seagreen")
plt.title("Ride Volume by Day of Week")
plt.xlabel("Day")
plt.ylabel("Number of Rides")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_ride_volume_by_weekday.png")
plt.close()

# ---------------------------------------------------------
# 3. Ride volume by city
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
bookings["city"].value_counts().plot(kind="bar", color="coral")
plt.title("Ride Volume by City")
plt.xlabel("City")
plt.ylabel("Number of Rides")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_ride_volume_by_city.png")
plt.close()

# ---------------------------------------------------------
# 4. Cancellation rate by city (heatmap-style bar)
# ---------------------------------------------------------
cancel_by_city = (
    bookings.groupby("city")["booking_status"]
    .apply(lambda x: (x == "Cancelled").mean() * 100)
    .sort_values(ascending=False)
)
plt.figure(figsize=(8, 5))
cancel_by_city.plot(kind="bar", color="firebrick")
plt.title("Cancellation Rate (%) by City")
plt.xlabel("City")
plt.ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_cancellation_rate_by_city.png")
plt.close()

# ---------------------------------------------------------
# 5. Cancellations by hour of day (heatmap: city x hour)
# ---------------------------------------------------------
pivot = bookings.pivot_table(
    index="city", columns="hour_of_day", values="booking_status",
    aggfunc=lambda x: (x == "Cancelled").mean() * 100
)
plt.figure(figsize=(14, 5))
sns.heatmap(pivot, cmap="Reds", annot=False)
plt.title("Cancellation Rate (%) Heatmap - City vs Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("City")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_cancellation_heatmap.png")
plt.close()

# ---------------------------------------------------------
# 6. Distance vs Fare correlation (scatter plot)
# ---------------------------------------------------------
sample = bookings.sample(3000, random_state=42)  # sample for a readable plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=sample, x="ride_distance_km", y="booking_value", alpha=0.4)
plt.title("Ride Distance vs Fare (Booking Value)")
plt.xlabel("Ride Distance (km)")
plt.ylabel("Booking Value (Fare)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_distance_vs_fare.png")
plt.close()

corr = bookings["ride_distance_km"].corr(bookings["booking_value"])
print(f"Correlation between distance and fare: {corr:.2f}")

# ---------------------------------------------------------
# 7. Driver rating distribution
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
drivers["avg_driver_rating"].plot(kind="hist", bins=20, color="mediumpurple")
plt.title("Driver Rating Distribution")
plt.xlabel("Average Driver Rating")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_driver_rating_distribution.png")
plt.close()

# ---------------------------------------------------------
# 8. Customer vs Driver rating comparison
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(customers["avg_customer_rating"], bins=20, alpha=0.5, label="Customer Rating")
plt.hist(drivers["avg_driver_rating"], bins=20, alpha=0.5, label="Driver Rating")
plt.legend()
plt.title("Customer vs Driver Rating Comparison")
plt.xlabel("Rating")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_customer_vs_driver_rating.png")
plt.close()

# ---------------------------------------------------------
# 9. Traffic / Weather vs Cancellation
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
cancel_by_traffic = (
    bookings.groupby("traffic_level")["booking_status"]
    .apply(lambda x: (x == "Cancelled").mean() * 100)
)
cancel_by_traffic.plot(kind="bar", color="darkorange")
plt.title("Cancellation Rate (%) by Traffic Level")
plt.ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/09_cancellation_by_traffic.png")
plt.close()

plt.figure(figsize=(8, 5))
cancel_by_weather = (
    bookings.groupby("weather_condition")["booking_status"]
    .apply(lambda x: (x == "Cancelled").mean() * 100)
)
cancel_by_weather.plot(kind="bar", color="teal")
plt.title("Cancellation Rate (%) by Weather Condition")
plt.ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/10_cancellation_by_weather.png")
plt.close()

# ---------------------------------------------------------
# 10. Payment/vehicle type usage patterns
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
bookings["vehicle_type"].value_counts().plot(kind="bar", color="slateblue")
plt.title("Vehicle Type Usage")
plt.ylabel("Number of Rides")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/11_vehicle_type_usage.png")
plt.close()

print("\nEDA complete. All charts saved inside the 'outputs/' folder.")
