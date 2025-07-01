import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load and clean data
df = pd.read_excel('Heat_Map_2025_05_21_to_2025_06_20.xlsx', sheet_name='EnergyConsumption').iloc[1:]
df.columns = ['date', 'hour', 'energy']
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str))
df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
df.dropna(subset=['datetime', 'energy'], inplace=True)
df = df[df['datetime'] < '2025-06-19']

# Feature engineering
df['hour_num'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Create dummy variables
hour_dummies = pd.get_dummies(df['hour_num'], prefix='hour')
dow_dummies = pd.get_dummies(df['day_of_week'], prefix='dow')

# Prepare for Prophet
df_prophet = pd.concat([df[['datetime', 'energy', 'is_weekend']], hour_dummies, dow_dummies], axis=1)
df_prophet.rename(columns={'datetime': 'ds', 'energy': 'y'}, inplace=True)

# Initialize and configure Prophet
model = Prophet(daily_seasonality=False, weekly_seasonality=False, changepoint_prior_scale=0.3)
model.add_regressor('is_weekend')
for col in hour_dummies.columns:
    model.add_regressor(col)
for col in dow_dummies.columns:
    model.add_regressor(col)

# Fit the model
model.fit(df_prophet)

# Prepare future dataframe for next 24 hours
future = model.make_future_dataframe(periods=24, freq='H')
future['hour_num'] = future['ds'].dt.hour
future['day_of_week'] = future['ds'].dt.dayofweek
future['is_weekend'] = future['day_of_week'].isin([5, 6]).astype(int)

# Create dummy variables for future
hour_dummies_future = pd.get_dummies(future['hour_num'], prefix='hour')
dow_dummies_future = pd.get_dummies(future['day_of_week'], prefix='dow')

# Align columns with training set
for col in hour_dummies.columns:
    if col not in hour_dummies_future:
        hour_dummies_future[col] = 0
hour_dummies_future = hour_dummies_future[hour_dummies.columns]

for col in dow_dummies.columns:
    if col not in dow_dummies_future:
        dow_dummies_future[col] = 0
dow_dummies_future = dow_dummies_future[dow_dummies.columns]

# Combine into final future df
future = pd.concat([future, hour_dummies_future, dow_dummies_future], axis=1)

# Forecast
forecast = model.predict(future)
forecast_june19 = forecast[(forecast['ds'] >= '2025-06-19') & (forecast['ds'] < '2025-06-20')]

# Plot forecast
plt.figure(figsize=(12, 6))
plt.bar(forecast_june19['ds'].dt.strftime('%H:%M'), forecast_june19['yhat'], color='dodgerblue')
plt.title('Forecast for June 19, 2025 (Using Hour, Day of Week, and Weekend)')
plt.xlabel('Hour')
plt.ylabel('Predicted Energy (kVAh)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

output_df = forecast_june19[['ds', 'yhat']].copy()
output_df.columns = ['datetime', 'predicted_energy_kVAh']
output_df['hour'] = output_df['datetime'].dt.strftime('%H:%M')
output_df = output_df[['datetime', 'hour', 'predicted_energy_kVAh']]
output_df.to_excel(r'C:\Users\IITMRP\PowerPrediction\prediction19.xlsx', index=False)
