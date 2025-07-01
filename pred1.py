import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel('Heat_Map_2025_05_21_to_2025_06_20.xlsx', sheet_name='EnergyConsumption').iloc[1:]
df.columns = ['date', 'hour', 'energy']
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str))
df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
df.dropna(subset=['datetime', 'energy'], inplace=True)
df = df[df['datetime'] < '2025-06-19']
df['hour_num'] = df['datetime'].dt.hour

# One-hot encode hour
hour_dummies = pd.get_dummies(df['hour_num'], prefix='hour')
df_prophet = pd.concat([df[['datetime', 'energy']], hour_dummies], axis=1).rename(columns={'datetime': 'ds', 'energy': 'y'})

# Initialize Prophet
model = Prophet(daily_seasonality=False, weekly_seasonality=False, changepoint_prior_scale=0.3)
for col in hour_dummies.columns:
    model.add_regressor(col)

# Fit the model
model.fit(df_prophet)

# Create future dataframe
future = model.make_future_dataframe(periods=24, freq='H')
future['hour_num'] = future['ds'].dt.hour

# One-hot encode future hours
future_hour_dummies = pd.get_dummies(future['hour_num'], prefix='hour')
for col in hour_dummies.columns:
    if col not in future_hour_dummies:
        future_hour_dummies[col] = 0
future_hour_dummies = future_hour_dummies[hour_dummies.columns]
future = pd.concat([future, future_hour_dummies], axis=1)

# Forecast
forecast = model.predict(future)
forecast_june19 = forecast[(forecast['ds'] >= '2025-06-19') & (forecast['ds'] < '2025-06-20')]

# Plot
plt.figure(figsize=(12, 6))
plt.bar(forecast_june19['ds'].dt.strftime('%H:%M'), forecast_june19['yhat'], color='skyblue')
plt.title('Hour-Wise Forecast (June 19, 2025) with Dips Modeled')
plt.xlabel('Hour')
plt.ylabel('Predicted Energy (kVAh)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
