from flask import Flask, render_template, jsonify
from collections import deque, Counter
import numpy as np
import joblib
import requests
import time
import csv
import os
from datetime import datetime, timedelta
import pandas as pd


# SETTINGS

WINDOW_SIZE = 4 
PHY_URL = "http://192.168.1.10:8080/get?accX&accY&accZ&gyrX&gyrY&gyrZ"

MODEL_PATH = "C:\\Users\\Dell\\Desktop\\AI Motion Sensor App\\ML model\\models\\activity_final_model.pkl"
ENCODER_PATH = "C:\\Users\\Dell\\Desktop\\AI Motion Sensor App\\ML model\\models\\label_encoder.pkl"
LOG_PATH = "activity_log.csv"

# Load model + encoder
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

print("✅ Model and encoder loaded")

# Buffers
accX_buffer = deque(maxlen=WINDOW_SIZE)
accY_buffer = deque(maxlen=WINDOW_SIZE)
accZ_buffer = deque(maxlen=WINDOW_SIZE)
gyroX_buffer = deque(maxlen=WINDOW_SIZE)
gyroY_buffer = deque(maxlen=WINDOW_SIZE)
gyroZ_buffer = deque(maxlen=WINDOW_SIZE)

app = Flask(__name__)


# FEATURE EXTRACTION

def extract_features():
    window = {
        "accX": np.array(accX_buffer),
        "accY": np.array(accY_buffer),
        "accZ": np.array(accZ_buffer),
        "gyrX": np.array(gyroX_buffer),
        "gyrY": np.array(gyroY_buffer),
        "gyrZ": np.array(gyroZ_buffer),
    }

    feats = {}
    for col, vals in window.items():
        feats[f"{col}_mean"] = np.mean(vals)
        feats[f"{col}_std"] = np.std(vals)
        feats[f"{col}_min"] = np.min(vals)
        feats[f"{col}_max"] = np.max(vals)
        feats[f"{col}_energy"] = np.sum(vals**2) / len(vals)

    return pd.DataFrame([feats])


# FETCH DATA FROM PHYPOX

def get_phyphox_data():
    try:
        response = requests.get(PHY_URL, timeout=1.0).json()
        accX = response['buffer']['accX']['buffer'][0] if response['buffer']['accX']['buffer'] else 0
        accY = response['buffer']['accY']['buffer'][0] if response['buffer']['accY']['buffer'] else 0
        accZ = response['buffer']['accZ']['buffer'][0] if response['buffer']['accZ']['buffer'] else 0
        gyroX = response['buffer']['gyrX']['buffer'][0] if response['buffer']['gyrX']['buffer'] else 0
        gyroY = response['buffer']['gyrY']['buffer'][0] if response['buffer']['gyrY']['buffer'] else 0
        gyroZ = response['buffer']['gyrZ']['buffer'][0] if response['buffer']['gyrZ']['buffer'] else 0
        return accX, accY, accZ, gyroX, gyroY, gyroZ
    except Exception as e:
        print("❌ Error fetching Phyphox data:", e)
        return None


# LOG PREDICTIONS

def log_prediction(activity, timestamp):
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "activity"])
        writer.writerow([timestamp, activity])


# REPORTS

def get_last_hour_stats():
    if not os.path.exists(LOG_PATH):
        return {}
    last_hour = datetime.now() - timedelta(hours=1)
    df = pd.read_csv(LOG_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df[df['timestamp'] >= last_hour]['activity'].value_counts().to_dict()

def get_today_stats():
    if not os.path.exists(LOG_PATH):
        return {}
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    df = pd.read_csv(LOG_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df[df['timestamp'] >= midnight]['activity'].value_counts().to_dict()

# Get timeline data for today
def get_daily_timeline():
    if not os.path.exists(LOG_PATH):
        return []
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    df = pd.read_csv(LOG_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    today_data = df[df['timestamp'] >= midnight]
    
    # Get the last 10 activities
    timeline_data = today_data.tail(10).to_dict('records')
    # Format the data for frontend
    result = []
    for item in timeline_data:
        result.append({
            'time': item['timestamp'].strftime('%H:%M:%S'),
            'activity': item['activity']
        })
    return result

# NEW: Get activity duration in minutes
def get_activity_duration():
    if not os.path.exists(LOG_PATH):
        return {}
    
    # Calculate time per sample (assuming 50Hz sampling rate)
    time_per_sample = 1/50  
    
    # Get today's data
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    df = pd.read_csv(LOG_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    today_data = df[df['timestamp'] >= midnight]
    
    # Calculate duration in minutes for each activity
    duration_minutes = {}
    for activity in today_data['activity'].unique():
        count = len(today_data[today_data['activity'] == activity])
        duration_minutes[activity] = round(count * time_per_sample / 60, 2)  # Convert to minutes
    
    return duration_minutes


# ROUTES

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_prediction")
def get_prediction():
    data = get_phyphox_data()
    prediction = "NO_DATA"

    if data:
        accX, accY, accZ, gyroX, gyroY, gyroZ = data
        accX_buffer.append(accX)
        accY_buffer.append(accY)
        accZ_buffer.append(accZ)
        gyroX_buffer.append(gyroX)
        gyroY_buffer.append(gyroY)
        gyroZ_buffer.append(gyroZ)

        if len(accX_buffer) == WINDOW_SIZE:
            try:
                feats = extract_features()
                y_pred = model.predict(feats)[0]
                prediction = encoder.inverse_transform([y_pred])[0]

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_prediction(prediction, timestamp)

                print(f"🎯 Prediction: {prediction}")
            except Exception as e:
                print("❌ Prediction error:", e)
                prediction = "ERROR"
        else:
            prediction = f"BUFFERING ({len(accX_buffer)}/{WINDOW_SIZE})"
    else:
        prediction = "NO_CONNECTION"

    timestamp = datetime.now().strftime("%H:%M:%S")
    return jsonify({"prediction": prediction, "time": timestamp})

@app.route("/get_last_hour_report")
def last_hour_report():
    return jsonify(get_activity_duration())  

@app.route("/get_daily_report")
def daily_report():
    return jsonify(get_activity_duration())  

# Timeline endpoint
@app.route("/get_daily_timeline")
def daily_timeline():
    return jsonify(get_daily_timeline())

if __name__ == "__main__":
    print("🚀 Flask server running with your RandomForest model")
    app.run(debug=True, host="0.0.0.0", port=5000)