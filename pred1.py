import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel('Heat_Map_2025_06_13_to_2025_06_27.xlsx', sheet_name='EnergyConsumption').iloc[1:]
df.columns = ['date', 'hour', 'energy']

# Combine date + hour
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str), errors='coerce')
df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
df.dropna(subset=['datetime', 'energy'], inplace=True)

# Train until June 27 inclusive
df = df[df['datetime'] <= '2025-06-27 23:59:59']
df['hour_num'] = df['datetime'].dt.hour

# One-hot encode hour
hour_dummies = pd.get_dummies(df['hour_num'], prefix='hour')
df_prophet = pd.concat([df[['datetime', 'energy']], hour_dummies], axis=1).rename(columns={'datetime': 'ds', 'energy': 'y'})

# Prophet model
model = Prophet(daily_seasonality=False, weekly_seasonality=False, changepoint_prior_scale=0.3)
for col in hour_dummies.columns:
    model.add_regressor(col)

model.fit(df_prophet)

# ---- FIXED FUTURE CREATION ----
# We want to forecast until June 28 end (23:00)
last_train_date = df['datetime'].max()
forecast_end = pd.to_datetime("2025-06-28 23:00:00")
horizon_hours = int((forecast_end - last_train_date).total_seconds() // 3600)

future = model.make_future_dataframe(periods=horizon_hours, freq='H')
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

# Extract June 28 forecast (full 00:00–23:00)
forecast_june28 = forecast[(forecast['ds'] >= '2025-06-28 00:00:00') & (forecast['ds'] <= '2025-06-28 23:00:00')]
# Convert to required format
output = pd.DataFrame({
    'date': forecast_june28['ds'].dt.date.astype(str),
    'hour': forecast_june28['ds'].dt.strftime('%H:%M:%S'),
    'energy': forecast_june28['yhat'].round(2)   # round to 2 decimals
})

output_df = forecast_june28[['ds', 'yhat']].copy()
output_df.columns = ['datetime', 'predicted_energy_kVAh']
output_df['hour'] = output_df['datetime'].dt.strftime('%H:%M')
output_df = output_df[['datetime', 'hour', 'predicted_energy_kVAh']]

# Save to Excel
output_file = r"C:\Users\HP\PowerPrediction\Forestcast_28.xlsx"
output_df.to_excel(output_file, index=False, engine="openpyxl")


print(f"✅ Forecast saved to {output_file}")

# Plot
plt.figure(figsize=(12, 6))
plt.bar(forecast_june28['ds'].dt.strftime('%H:%M'), forecast_june28['yhat'], color='skyblue')
plt.title('Hour-Wise Forecast (June 28, 2025)')
plt.xlabel('Hour')
plt.ylabel('Predicted Energy (kVAh)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
