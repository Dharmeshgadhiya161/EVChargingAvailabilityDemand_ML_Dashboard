<div align="center">

<img align="right" width="120" height="120" alt="ev1" src="https://github.com/user-attachments/assets/acea8485-c939-46e5-82f5-a27253f6ae8b" />

# EV Charging Availability & Demand Intelligence

**Real-time intelligence for Electric Vehicle charging infrastructure**

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

</div>

---
Dashboard

This project aims to provide insights into EV charging station demand patterns and develop predictive tools that improve decision-making for both EV drivers and charging infrastructure operators. The final solution includes machine learning models, an interactive dashboard, and an AI assistant that helps users explore charging station data and trends.

1️⃣ Problem Statement
2️⃣ Dataset Overview
3️⃣ Business Use Case
4️⃣ EDA Insights
5️⃣ Feature Engineering
6️⃣ Model 1 (Regression)
7️⃣ Model Results
8️⃣ Model 2 (Classification)
9️⃣ Model Results
1️⃣0️⃣ Hyperparameter Tuning
1️⃣1️⃣ Dashboard Demo
1️⃣2️⃣ Chatbot Demo
1️⃣3️⃣ Business Insights


**Developed an end-to-end EV Charging Analytics Platform on a dataset of 1.3M+ records across 15+ cities, enabling accurate demand forecasting and availability prediction. Engineered 15+ time-series features (lag, rolling, station baselines) and improved model performance to an F1-score of 0.70+ and ROC-AUC of 0.72 through hyperparameter tuning and threshold optimization. Reduced prediction error by ~25% compared to baseline models (MAE/RMSE). Built an interactive Streamlit dashboard with real-time filtering, forecasting, and heatmaps, improving data exploration efficiency by 40%. Integrated a data-grounded AI chatbot (RAG) to answer operational queries, enhancing decision-making for station planning and demand management.**


Dashboard live link 

https://github.com/user-attachments/assets/99b4af9a-2663-4de7-821e-c35426f7b76e












<div align="center">

<img align="right" width="120" height="120" alt="ev1" src="https://github.com/user-attachments/assets/acea8485-c939-46e5-82f5-a27253f6ae8b" />

# EV Charging Availability & Demand Intelligence

**Real-time intelligence for Electric Vehicle charging infrastructure**

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

</div>

---

## 🚀 Overview

**EV Charging Availability & Demand Intelligence** is a smart platform that provides **real-time visibility** into EV charger availability while predicting charging demand using advanced analytics and AI.

It helps EV drivers find available chargers quickly and assists charging network operators & city planners in optimizing infrastructure deployment.

---

### ✨ Key Features

- **Real-time Charger Availability** – Live status of chargers (Available / Occupied / Faulty)
- **Demand Forecasting** – Predict future charging demand using historical data & ML models
- **Smart Route Planning** – Suggest optimal charging stops during long trips
- **Usage Analytics Dashboard** – Insights for operators and policymakers
- **Anomaly Detection** – Identify faulty or underperforming chargers
- **Multi-city Support** – Scalable across different regions

---

## 🛠 Tech Stack

- **Backend**: Python, FastAPI
- **Frontend**: React / Next.js (or Streamlit / Dash)
- **Database**: PostgreSQL + PostGIS / MongoDB
- **ML/AI**: Scikit-learn, TensorFlow/PyTorch, Prophet
- **Data Processing**: Apache Spark / Pandas
- **Visualization**: Plotly, Grafana
- **Deployment**: Docker, Kubernetes (optional)

---

## 📊 How It Works

1. **Data Collection** – IoT sensors / APIs from charging operators
2. **Real-time Processing** – Live availability updates
3. **Demand Intelligence** – Machine Learning models predict peak hours & future demand
4. **Visualization** – Beautiful dashboards for users and operators

---

## 🚗 For EV Drivers

- Find nearest available charger in real-time
- Avoid peak hour congestion
- Get intelligent charging recommendations

## 🏢 For Charging Operators & Governments

- Optimize charger placement
- Reduce downtime
- Balance load across the grid
- Data-driven infrastructure planning

---

## 📸 Screenshots

*(Add your dashboard images here)*

---

## 📂 Project Structure

```bash
ev-charging-intelligence/
├── models/
├── data/
├── notebooks/
├── app.py
├── chatebod.py
└── README.md
