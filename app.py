import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import time


# -------------------------------
# Simulate Sensor Data Function
# -------------------------------
def generate_sensor_data():
    return {
        'battery_voltage': np.random.normal(48, 2),
        'motor_temp': np.random.normal(50, 5),
        'current_draw': np.random.normal(10, 2),
        'vibration_level': np.random.normal(0.2, 0.05),
    }


# -------------------------------
# Initial Training Data (Normal)
# -------------------------------
st.title("🔍 Real-Time E-Bike Anomaly Detection")
data = pd.DataFrame(columns=['battery_voltage', 'motor_temp', 'current_draw', 'vibration_level'])

for _ in range(100):
    new_row = pd.DataFrame([generate_sensor_data()])
    data = pd.concat([data, new_row], ignore_index=True)

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(data)

# Streamlit placeholders
status_placeholder = st.empty()
table_placeholder = st.empty()
chart_placeholder = st.empty()

# Store live history
live_data = pd.DataFrame(columns=data.columns.tolist() + ['Status'])

# -------------------------------
# Real-Time Simulation Loop
# -------------------------------
while True:
    new_data = pd.DataFrame([generate_sensor_data()])
    prediction = model.predict(new_data)[0]
    status = "Anomaly ❌" if prediction == -1 else "Normal ✅"

    # Add status to new row
    new_data['Status'] = status
    live_data = pd.concat([live_data, new_data], ignore_index=True)

    # Keep only last 50 entries
    if len(live_data) > 50:
        live_data = live_data.tail(50)

    # Update dashboard
    status_placeholder.markdown(f"## Current Status: **{status}**")
    table_placeholder.dataframe(live_data.tail(10).reset_index(drop=True))

    chart_placeholder.line_chart(live_data.drop(columns=['Status']))

    time.sleep(1)
