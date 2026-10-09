import streamlit as st
import pickle
import numpy as np
import requests
from datetime import datetime

# Load trained model
model = pickle.load(open("bike_model.pkl", "rb"))

st.title("🚴 Seoul Bike Rental Prediction (Real-Time Weather)")
st.write("Weather values are automatically fetched from Seoul's live weather API.")

# ----------------------------------------
# FETCH LIVE SEOUL WEATHER
# ----------------------------------------
def fetch_seoul_weather():
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=37.5665&longitude=126.9780&"
        "current=temperature_2m,dewpoint_2m,relative_humidity_2m,"
        "uv_index,wind_speed_10m"
    )
    
    response = requests.get(url)
    data = response.json()

    temp = data["current"]["temperature_2m"]
    dew = data["current"]["dewpoint_2m"]
    humidity = data["current"]["relative_humidity_2m"]
    wind = data["current"]["wind_speed_10m"]
    uv_index = data["current"]["uv_index"]

    # Approximations
    solar_radiation = uv_index * 0.04
    visibility = 200     # dataset scale
    rainfall = 0         # Open-Meteo does not give current rainfall in mm
    snowfall = 0         # dataset mostly zero
    
    return temp, dew, humidity, wind, solar_radiation, visibility, rainfall, snowfall


st.subheader("📡 Fetching Real-Time Seoul Weather...")
temp, dew, humidity, wind, solar, visibility, rainfall, snowfall = fetch_seoul_weather()
st.success("✅ Weather loaded successfully!")


# ----------------------------------------
# USER INPUTS (Non-weather)
# ----------------------------------------
hour = st.number_input("Hour (0–23)", min_value=0, max_value=23, value=datetime.now().hour)
holiday = st.selectbox("Holiday?", ["No Holiday", "Holiday"])
func_day = st.selectbox("Functioning Day?", ["Yes", "No"])
season = st.selectbox("Season", ["Autumn", "Spring", "Summer", "Winter"])

# ----------------------------------------
# ENCODING
# ----------------------------------------
holiday_code = 1 if holiday == "Holiday" else 0
function_code = 1 if func_day == "Yes" else 0

season_map = {
    "Autumn": [1, 0, 0, 0],
    "Spring": [0, 1, 0, 0],
    "Summer": [0, 0, 1, 0],
    "Winter": [0, 0, 0, 1]
}
season_encoded = season_map[season]

year = 2018  # dataset year used during training

# ----------------------------------------
# PREPARE INPUT ARRAY
# Same order as train_model.py (VERY IMPORTANT)
# ----------------------------------------
input_data = np.array([[
    hour,
    temp,
    humidity,
    wind,
    visibility,
    dew,
    solar,
    rainfall,
    snowfall,
    holiday_code,
    function_code,
    season_encoded[0],  # Autumn
    season_encoded[1],  # Spring
    season_encoded[2],  # Summer
    season_encoded[3],  # Winter
    year
]])

# ----------------------------------------
# SHOW WEATHER VALUES
# ----------------------------------------
st.subheader("🌤 Live Seoul Weather Data")

st.write(f"**Temperature:** {temp} °C")
st.write(f"**Dew Point:** {dew} °C")
st.write(f"**Humidity:** {humidity}%")
st.write(f"**Wind Speed:** {wind} m/s")
st.write(f"**Solar Radiation (approx):** {solar:.3f} MJ/m²")
st.write(f"**Visibility:** {visibility}")
st.write(f"**Rainfall:** {rainfall} mm")
st.write(f"**Snowfall:** {snowfall} cm")

# ----------------------------------------
# PREDICT
# ----------------------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Estimated Bike Rentals: {int(prediction)}")