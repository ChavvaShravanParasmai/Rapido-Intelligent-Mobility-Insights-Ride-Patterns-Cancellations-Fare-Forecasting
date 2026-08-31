"""
STEP 4 - MODEL TRAINING
------------------------
We train four simple, beginner-friendly Machine Learning models
using the RandomForest algorithm (easy to use and gives good
results without heavy tuning):

MODEL 1: Ride Outcome Prediction (Multi-class Classification)
    Predicts whether a ride will be Completed / Cancelled / Incomplete
    BEFORE the ride happens (so we only use info known at booking time).

MODEL 2: Fare Prediction (Regression)
    Predicts the booking_value (fare) using distance, traffic,
    weather, time, vehicle type, and surge.

MODEL 3: Customer Cancellation Risk (Binary Classification)
    Predicts the probability that a customer will cancel a booking,
    using their historical cancellation rate, rating, and behaviour.

MODEL 4: Driver Delay Prediction (Binary Classification)
    Predicts whether a driver is likely to cause a delay, using
    their delay history, acceptance rate, and current traffic level.

All trained models are saved into the 'models/' folder using
joblib, so the Streamlit app can load and reuse them later.
"""

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv("data/cleaned/master_dataset.csv")

# ===========================================================
# MODEL 1 - RIDE OUTCOME PREDICTION (Classification)
# ===========================================================
print("=" * 60)
print("MODEL 1: Ride Outcome Prediction (Completed/Cancelled/Incomplete)")
print("=" * 60)

# We only use features that are known BEFORE the ride starts.
# (we do NOT use actual_ride_time_min or fare_per_min because those
# depend on the ride actually happening)
outcome_features = [
    "hour_of_day", "is_weekend", "rush_hour_flag", "long_distance_flag",
    "city", "vehicle_type", "traffic_level", "weather_condition",
    "ride_distance_km", "estimated_ride_time_min", "base_fare",
    "surge_multiplier", "customer_loyalty_score", "cancellation_rate",
    "driver_reliability_score", "delay_rate",
]

df_outcome = df[outcome_features + ["booking_status"]].copy()

# Encode text (categorical) columns into numbers using LabelEncoder
cat_cols_outcome = ["city", "vehicle_type", "traffic_level", "weather_condition"]
encoders_outcome = {}
for col in cat_cols_outcome:
    le = LabelEncoder()
    df_outcome[col] = le.fit_transform(df_outcome[col])
    encoders_outcome[col] = le

# Encode the target column (booking_status)
target_encoder = LabelEncoder()
df_outcome["booking_status_encoded"] = target_encoder.fit_transform(df_outcome["booking_status"])

X = df_outcome[outcome_features]
y = df_outcome["booking_status_encoded"]

# Split into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a Random Forest Classifier (simple, no heavy tuning needed)
clf = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_leaf=20, random_state=42, n_jobs=-1
)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score (weighted): {f1:.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance (which features matter most)
importance_outcome = pd.Series(clf.feature_importances_, index=outcome_features).sort_values(ascending=False)
print("\nTop features for Ride Outcome model:")
print(importance_outcome.head(8))

# Save model + encoders
joblib.dump(clf, f"{MODEL_DIR}/ride_outcome_model.pkl", compress=3)
joblib.dump(encoders_outcome, f"{MODEL_DIR}/ride_outcome_encoders.pkl")
joblib.dump(target_encoder, f"{MODEL_DIR}/ride_outcome_target_encoder.pkl")

# ===========================================================
# MODEL 2 - FARE PREDICTION (Regression)
# ===========================================================
print("\n" + "=" * 60)
print("MODEL 2: Fare Prediction (Regression)")
print("=" * 60)

fare_features = [
    "ride_distance_km", "estimated_ride_time_min", "hour_of_day",
    "is_weekend", "rush_hour_flag", "long_distance_flag",
    "city", "vehicle_type", "traffic_level", "weather_condition",
    "surge_multiplier",
]

df_fare = df[fare_features + ["booking_value"]].copy()

cat_cols_fare = ["city", "vehicle_type", "traffic_level", "weather_condition"]
encoders_fare = {}
for col in cat_cols_fare:
    le = LabelEncoder()
    df_fare[col] = le.fit_transform(df_fare[col])
    encoders_fare[col] = le

X = df_fare[fare_features]
y = df_fare["booking_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_leaf=20, random_state=42, n_jobs=-1
)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

# RMSE as a percentage of the average fare, to compare with the
# project's target benchmark (RMSE within +/-10% of actual fare)
avg_fare = y_test.mean()
rmse_pct = (rmse / avg_fare) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 score: {r2:.3f}")
print(f"RMSE as % of average fare: {rmse_pct:.1f}%  (target: within 10%)")

importance_fare = pd.Series(reg.feature_importances_, index=fare_features).sort_values(ascending=False)
print("\nTop features for Fare Prediction model:")
print(importance_fare.head(8))

joblib.dump(reg, f"{MODEL_DIR}/fare_prediction_model.pkl", compress=3)
joblib.dump(encoders_fare, f"{MODEL_DIR}/fare_prediction_encoders.pkl")

# ===========================================================
# MODEL 3 - CUSTOMER CANCELLATION RISK (Binary Classification)
# ===========================================================
print("\n" + "=" * 60)
print("MODEL 3: Customer Cancellation Risk Prediction")
print("=" * 60)

# IMPORTANT: customer_cancel_flag turns out to be a simple rule
# (cancellation_rate > 0.2 -> flag = 1) baked into this dataset.
# So we deliberately EXCLUDE cancellation_rate, customer_loyalty_score,
# and completion_rate from the inputs — using them would mean the
# model just "looks up the answer" instead of learning a real pattern.
# We only use genuinely independent signals: rating, tenure, booking
# count, and the ride's own context (time/surge).
cancel_features = [
    "avg_customer_rating", "total_bookings", "customer_signup_days_ago",
    "rush_hour_flag", "surge_multiplier", "hour_of_day", "is_weekend",
    "ride_distance_km", "traffic_level_encoded",
]

df_cancel = df[["avg_customer_rating", "total_bookings", "customer_signup_days_ago",
                 "rush_hour_flag", "surge_multiplier", "hour_of_day", "is_weekend",
                 "ride_distance_km", "traffic_level", "customer_cancel_flag"]].copy()

le_traffic_cancel = LabelEncoder()
df_cancel["traffic_level_encoded"] = le_traffic_cancel.fit_transform(df_cancel["traffic_level"])

X = df_cancel[cancel_features]
y = df_cancel["customer_cancel_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cancel_clf = RandomForestClassifier(
    n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1
)
cancel_clf.fit(X_train, y_train)

y_pred = cancel_clf.predict(X_test)
y_proba = cancel_clf.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"AUC: {auc:.3f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

importance_cancel = pd.Series(cancel_clf.feature_importances_, index=cancel_features).sort_values(ascending=False)
print("\nTop features for Cancellation Risk model:")
print(importance_cancel.head(6))

joblib.dump(cancel_clf, f"{MODEL_DIR}/cancellation_risk_model.pkl", compress=3)
joblib.dump({"traffic_level": le_traffic_cancel}, f"{MODEL_DIR}/cancellation_risk_encoders.pkl")

# ===========================================================
# MODEL 4 - DRIVER DELAY PREDICTION (Binary Classification)
# ===========================================================
print("\n" + "=" * 60)
print("MODEL 4: Driver Delay Prediction")
print("=" * 60)

# Same reasoning as Model 3: driver_delay_flag is a simple rule
# (delay_rate > 0.1 -> flag = 1). We exclude delay_rate and
# driver_reliability_score (which is built from delay_rate) from
# the inputs, and instead use acceptance rate, experience, rating,
# and the ride's own traffic conditions.
delay_features = [
    "acceptance_rate", "avg_driver_rating", "driver_experience_years",
    "total_assigned_rides", "traffic_level_encoded", "hour_of_day",
    "rush_hour_flag",
]

df_delay = df[["acceptance_rate", "avg_driver_rating", "driver_experience_years",
                "total_assigned_rides", "traffic_level", "hour_of_day",
                "rush_hour_flag", "driver_delay_flag"]].copy()

le_traffic_delay = LabelEncoder()
df_delay["traffic_level_encoded"] = le_traffic_delay.fit_transform(df_delay["traffic_level"])

X = df_delay[delay_features]
y = df_delay["driver_delay_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

delay_clf = RandomForestClassifier(
    n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=42,
    class_weight="balanced", n_jobs=-1
)
delay_clf.fit(X_train, y_train)

y_pred = delay_clf.predict(X_test)
y_proba = delay_clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"AUC: {auc:.3f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

importance_delay = pd.Series(delay_clf.feature_importances_, index=delay_features).sort_values(ascending=False)
print("\nTop features for Driver Delay model:")
print(importance_delay.head(6))

joblib.dump(delay_clf, f"{MODEL_DIR}/driver_delay_model.pkl", compress=3)
joblib.dump({"traffic_level": le_traffic_delay}, f"{MODEL_DIR}/driver_delay_encoders.pkl")

print("\nAll 4 models trained and saved in the 'models/' folder.")
