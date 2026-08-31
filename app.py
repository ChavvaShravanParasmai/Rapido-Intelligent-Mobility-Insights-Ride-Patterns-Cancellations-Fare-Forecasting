"""
STEP 6 - STREAMLIT DASHBOARD
------------------------------
A simple, interactive dashboard for the Rapido Mobility Insights
project. It has 2 pages (selectable from the sidebar):

1. Dashboard  -> filters + charts + KPIs (business insights)
2. Predict a Ride -> uses our trained ML models to predict the
   ride outcome (Completed/Cancelled/Incomplete) and the fare
   for a new ride you describe.

To run this app:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(page_title="Rapido Mobility Insights", layout="wide")
st.title("🛵 Rapido: Intelligent Mobility Insights")
st.caption("Ride Patterns, Cancellations & Fare Forecasting")


# ---------------------------------------------------------
# Load data (cached so it's fast after the first load)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned/master_dataset.csv")
    df["booking_datetime"] = pd.to_datetime(df["booking_datetime"])
    return df


@st.cache_resource
def load_models():
    outcome_model = joblib.load("models/ride_outcome_model.pkl")
    outcome_encoders = joblib.load("models/ride_outcome_encoders.pkl")
    outcome_target_encoder = joblib.load("models/ride_outcome_target_encoder.pkl")
    fare_model = joblib.load("models/fare_prediction_model.pkl")
    fare_encoders = joblib.load("models/fare_prediction_encoders.pkl")
    cancel_model = joblib.load("models/cancellation_risk_model.pkl")
    cancel_encoders = joblib.load("models/cancellation_risk_encoders.pkl")
    delay_model = joblib.load("models/driver_delay_model.pkl")
    delay_encoders = joblib.load("models/driver_delay_encoders.pkl")
    return (outcome_model, outcome_encoders, outcome_target_encoder, fare_model,
            fare_encoders, cancel_model, cancel_encoders, delay_model, delay_encoders)


df = load_data()
(outcome_model, outcome_encoders, outcome_target_encoder, fare_model,
 fare_encoders, cancel_model, cancel_encoders, delay_model, delay_encoders) = load_models()

# ---------------------------------------------------------
# Sidebar: choose page
# ---------------------------------------------------------
page = st.sidebar.radio(
    "Go to",
    ["📊 Dashboard", "🔮 Predict a Ride", "🙋 Customer Cancellation Risk", "🚗 Driver Delay Risk"]
)

# ===========================================================
# PAGE 1: DASHBOARD
# ===========================================================
if page == "📊 Dashboard":

    # ---- Filters ----
    st.sidebar.header("Filters")
    cities = st.sidebar.multiselect(
        "City", options=sorted(df["city"].unique()), default=sorted(df["city"].unique())
    )
    vehicle_types = st.sidebar.multiselect(
        "Vehicle Type", options=sorted(df["vehicle_type"].unique()), default=sorted(df["vehicle_type"].unique())
    )

    filtered = df[df["city"].isin(cities) & df["vehicle_type"].isin(vehicle_types)]

    # ---- KPI cards ----
    col1, col2, col3, col4 = st.columns(4)
    total_rides = len(filtered)
    cancel_rate = (filtered["booking_status"] == "Cancelled").mean() * 100
    avg_fare = filtered["booking_value"].mean()
    avg_distance = filtered["ride_distance_km"].mean()

    col1.metric("Total Rides", f"{total_rides:,}")
    col2.metric("Cancellation Rate", f"{cancel_rate:.1f}%")
    col3.metric("Average Fare", f"₹{avg_fare:.0f}")
    col4.metric("Average Distance", f"{avg_distance:.1f} km")

    st.divider()

    # ---- Charts ----
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Ride Volume by Hour of Day")
        hourly = filtered["hour_of_day"].value_counts().sort_index()
        fig = px.bar(x=hourly.index, y=hourly.values, labels={"x": "Hour", "y": "Rides"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Booking Status Breakdown")
        status_counts = filtered["booking_status"].value_counts()
        fig = px.pie(names=status_counts.index, values=status_counts.values)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Cancellation Rate by City")
        cancel_city = (
            filtered.groupby("city")["booking_status"]
            .apply(lambda x: (x == "Cancelled").mean() * 100)
            .sort_values(ascending=False)
        )
        fig = px.bar(x=cancel_city.index, y=cancel_city.values, labels={"x": "City", "y": "Cancellation %"})
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Average Fare by Vehicle Type")
        fare_vehicle = filtered.groupby("vehicle_type")["booking_value"].mean().sort_values(ascending=False)
        fig = px.bar(x=fare_vehicle.index, y=fare_vehicle.values, labels={"x": "Vehicle Type", "y": "Avg Fare"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cancellation Rate Heatmap: City vs Hour of Day")
    pivot = filtered.pivot_table(
        index="city", columns="hour_of_day", values="booking_status",
        aggfunc=lambda x: (x == "Cancelled").mean() * 100
    )
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Reds",
                     labels=dict(x="Hour of Day", y="City", color="Cancel %"))
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================
# PAGE 2: PREDICT A RIDE
# ===========================================================
elif page == "🔮 Predict a Ride":
    st.header("🔮 Predict Ride Outcome & Fare")
    st.write("Fill in the ride details below to get a prediction.")

    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", sorted(df["city"].unique()))
        vehicle_type = st.selectbox("Vehicle Type", sorted(df["vehicle_type"].unique()))
        traffic_level = st.selectbox("Traffic Level", sorted(df["traffic_level"].unique()))
        weather_condition = st.selectbox("Weather Condition", sorted(df["weather_condition"].unique()))
        hour_of_day = st.slider("Hour of Day", 0, 23, 9)

    with col2:
        ride_distance_km = st.number_input("Ride Distance (km)", min_value=0.5, max_value=100.0, value=8.0)
        estimated_ride_time_min = st.number_input("Estimated Ride Time (min)", min_value=1.0, max_value=180.0, value=25.0)
        base_fare = st.number_input("Base Fare (₹)", min_value=10.0, max_value=1000.0, value=80.0)
        surge_multiplier = st.slider("Surge Multiplier", 1.0, 3.0, 1.2, step=0.1)
        is_weekend = st.checkbox("Is it a weekend?")

    rush_hour_flag = 1 if (8 <= hour_of_day <= 10) or (17 <= hour_of_day <= 20) else 0
    long_distance_flag = 1 if ride_distance_km > 15 else 0

    # Use overall dataset averages as reasonable defaults for
    # customer/driver behaviour features (in a real product these
    # would come from the specific customer/driver's history).
    avg_customer_loyalty = df["customer_loyalty_score"].mean()
    avg_cancellation_rate = df["cancellation_rate"].mean()
    avg_driver_reliability = df["driver_reliability_score"].mean()
    avg_delay_rate = df["delay_rate"].mean()

    if st.button("Predict"):
        # ---- Predict Ride Outcome ----
        outcome_input = pd.DataFrame([{
            "hour_of_day": hour_of_day,
            "is_weekend": int(is_weekend),
            "rush_hour_flag": rush_hour_flag,
            "long_distance_flag": long_distance_flag,
            "city": outcome_encoders["city"].transform([city])[0],
            "vehicle_type": outcome_encoders["vehicle_type"].transform([vehicle_type])[0],
            "traffic_level": outcome_encoders["traffic_level"].transform([traffic_level])[0],
            "weather_condition": outcome_encoders["weather_condition"].transform([weather_condition])[0],
            "ride_distance_km": ride_distance_km,
            "estimated_ride_time_min": estimated_ride_time_min,
            "base_fare": base_fare,
            "surge_multiplier": surge_multiplier,
            "customer_loyalty_score": avg_customer_loyalty,
            "cancellation_rate": avg_cancellation_rate,
            "driver_reliability_score": avg_driver_reliability,
            "delay_rate": avg_delay_rate,
        }])

        outcome_pred = outcome_model.predict(outcome_input)[0]
        outcome_label = outcome_target_encoder.inverse_transform([outcome_pred])[0]
        outcome_proba = outcome_model.predict_proba(outcome_input)[0]

        # ---- Predict Fare ----
        fare_input = pd.DataFrame([{
            "ride_distance_km": ride_distance_km,
            "estimated_ride_time_min": estimated_ride_time_min,
            "hour_of_day": hour_of_day,
            "is_weekend": int(is_weekend),
            "rush_hour_flag": rush_hour_flag,
            "long_distance_flag": long_distance_flag,
            "city": fare_encoders["city"].transform([city])[0],
            "vehicle_type": fare_encoders["vehicle_type"].transform([vehicle_type])[0],
            "traffic_level": fare_encoders["traffic_level"].transform([traffic_level])[0],
            "weather_condition": fare_encoders["weather_condition"].transform([weather_condition])[0],
            "surge_multiplier": surge_multiplier,
        }])

        fare_pred = fare_model.predict(fare_input)[0]

        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.subheader("Predicted Ride Outcome")
            if outcome_label == "Completed":
                st.success(f"✅ {outcome_label}")
            elif outcome_label == "Cancelled":
                st.error(f"❌ {outcome_label}")
            else:
                st.warning(f"⚠️ {outcome_label}")

            proba_df = pd.DataFrame({
                "Outcome": outcome_target_encoder.classes_,
                "Probability": outcome_proba
            }).sort_values("Probability", ascending=False)
            st.dataframe(proba_df, hide_index=True)

        with r2:
            st.subheader("Predicted Fare")
            st.metric("Estimated Fare", f"₹{fare_pred:.2f}")

# ===========================================================
# PAGE 3: CUSTOMER CANCELLATION RISK
# ===========================================================
elif page == "🙋 Customer Cancellation Risk":
    st.header("🙋 Customer Cancellation Risk")
    st.write(
        "Estimate how likely a customer is to cancel a ride, based on "
        "their profile and the ride's context."
    )

    col1, col2 = st.columns(2)
    with col1:
        avg_customer_rating = st.slider("Average Customer Rating", 1.0, 5.0, 4.2, step=0.1)
        total_bookings = st.number_input("Total Past Bookings", min_value=0, max_value=500, value=20)
        customer_signup_days_ago = st.number_input("Days Since Signup", min_value=0, max_value=3000, value=180)
        traffic_level = st.selectbox(
            "Traffic Level", sorted(df["traffic_level"].unique()), key="cancel_traffic"
        )

    with col2:
        hour_of_day = st.slider("Hour of Day", 0, 23, 9, key="cancel_hour")
        is_weekend = st.checkbox("Is it a weekend?", key="cancel_weekend")
        surge_multiplier = st.slider("Surge Multiplier", 1.0, 3.0, 1.2, step=0.1, key="cancel_surge")
        ride_distance_km = st.number_input(
            "Ride Distance (km)", min_value=0.5, max_value=100.0, value=8.0, key="cancel_distance"
        )

    rush_hour_flag = 1 if (8 <= hour_of_day <= 10) or (17 <= hour_of_day <= 20) else 0

    if st.button("Check Cancellation Risk"):
        cancel_input = pd.DataFrame([{
            "avg_customer_rating": avg_customer_rating,
            "total_bookings": total_bookings,
            "customer_signup_days_ago": customer_signup_days_ago,
            "rush_hour_flag": rush_hour_flag,
            "surge_multiplier": surge_multiplier,
            "hour_of_day": hour_of_day,
            "is_weekend": int(is_weekend),
            "ride_distance_km": ride_distance_km,
            "traffic_level_encoded": cancel_encoders["traffic_level"].transform([traffic_level])[0],
        }])

        risk_proba = cancel_model.predict_proba(cancel_input)[0][1]

        st.divider()
        st.subheader("Cancellation Risk Result")
        st.metric("Cancellation Probability", f"{risk_proba * 100:.1f}%")
        if risk_proba >= 0.6:
            st.error("⚠️ High risk of cancellation — consider a confirmation call or backup driver.")
        elif risk_proba >= 0.4:
            st.warning("🟠 Moderate risk of cancellation.")
        else:
            st.success("✅ Low risk of cancellation.")

# ===========================================================
# PAGE 4: DRIVER DELAY RISK
# ===========================================================
else:
    st.header("🚗 Driver Delay Risk")
    st.write(
        "Estimate how likely a driver is to cause a delay or incomplete "
        "ride, based on their performance history and current conditions."
    )

    col1, col2 = st.columns(2)
    with col1:
        acceptance_rate = st.slider("Driver Acceptance Rate", 0.0, 1.0, 0.85, step=0.01)
        avg_driver_rating = st.slider("Average Driver Rating", 1.0, 5.0, 4.3, step=0.1)
        driver_experience_years = st.number_input("Driver Experience (years)", min_value=0.0, max_value=30.0, value=2.0)

    with col2:
        total_assigned_rides = st.number_input("Total Assigned Rides", min_value=0, max_value=20000, value=500)
        traffic_level = st.selectbox(
            "Traffic Level", sorted(df["traffic_level"].unique()), key="delay_traffic"
        )
        hour_of_day = st.slider("Hour of Day", 0, 23, 9, key="delay_hour")

    rush_hour_flag = 1 if (8 <= hour_of_day <= 10) or (17 <= hour_of_day <= 20) else 0

    if st.button("Check Delay Risk"):
        delay_input = pd.DataFrame([{
            "acceptance_rate": acceptance_rate,
            "avg_driver_rating": avg_driver_rating,
            "driver_experience_years": driver_experience_years,
            "total_assigned_rides": total_assigned_rides,
            "traffic_level_encoded": delay_encoders["traffic_level"].transform([traffic_level])[0],
            "hour_of_day": hour_of_day,
            "rush_hour_flag": rush_hour_flag,
        }])

        delay_proba = delay_model.predict_proba(delay_input)[0][1]

        st.divider()
        st.subheader("Delay Risk Result")
        st.metric("Delay Probability", f"{delay_proba * 100:.1f}%")
        if delay_proba >= 0.5:
            st.error("⚠️ High risk of delay — consider reassigning to a more reliable driver.")
        elif delay_proba >= 0.3:
            st.warning("🟠 Moderate risk of delay.")
        else:
            st.success("✅ Low risk of delay.")
