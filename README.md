# AI Motion Detection System  

## 📌 Overview  
This project was developed for the **AI Motion Sensor Competition**, organized by **IEEE ComSoc and MathWorks**.  

We built a **real-time human activity recognition system** using smartphone sensor data (accelerometer & gyroscope).  
The system detects activities such as **Sitting, Standing, Walking, and Lying Down** in real time and visualizes them on a **Flask web application** with a live dashboard.  

---

## 👥 Team  
- Mohammad Al-Hriry
- Oamr Elmahrouk
- Ra’ad Hawamdeh
  
---

## 🎯 Project Workflow  

### 1️⃣ Data Collection  
- Data was recorded using the **Phyphox app** on a smartphone.  
- Activities captured: Sitting, Standing, Walking, Lying down.  
- Each activity recorded for ~10 minutes.  
- Sensors used: **Accelerometer + Gyroscope**.  

### 2️⃣ Feature Extraction  
- Sliding window approach (100 samples per window, 50% overlap).  
- Extracted **30 statistical features** (mean, std, min, max, energy) from each window.  
- Tools used: **Python, Pandas, NumPy**.  

### 3️⃣ Model Training  
- Trained a **Random Forest Classifier**.  
- Achieved **99.7% accuracy** with strong precision, recall, and F1-score.  
- Model saved using **joblib** for deployment.  

### 4️⃣ Flask Web Application  
- **Backend (Flask/Python)**: real-time data fetching, feature extraction, ML prediction.  
- **Frontend (HTML/CSS/JS)**: responsive dashboard with live updates.  
- **Features**:  
  - Real-time prediction of current activity  
  - Last-hour and daily activity reports  
  - Goal tracking (e.g., daily walking goal)  

### 5️⃣ Testing  
- System tested in real-time with Phyphox streaming data.  
- Predictions updated every ~2 seconds.  
- Smooth performance (<100ms processing per prediction).  

---

## 🖥️ Demo & Presentation  
- [1-Minute Demo Video (Google Drive)](YOUR_DEMO_LINK_HERE)  
- [3-Minute Presentation Video (Google Drive)](YOUR_PRESENTATION_LINK_HERE)  

---
├── data/                  # Raw collected data
├── feature_extraction.py  # Feature extraction script
├── features/              # Extracted features
├── models/                # Trained ML models (.pkl files)

├── templates/             # HTML frontend
├── app.py                 # Flask backend

├── README.md              # Project documentation

### 📊 Results

- Accuracy: 99.7%

- Real-time prediction speed: <100ms

- Handles 50Hz sensor data seamlessly

- User-friendly web dashboard

### 📌 Future Work

- Add more activities (Running, Cycling, Stairs).

- Deploy as a mobile app.

- Add cloud integration for long-term storage.

- Multi-user tracking and analytics.

### 🏆 Competition

- This project was submitted to the AI Motion Sensor Competition (2025), organized by IEEE ComSoc and MathWorks.

