import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np

# === Load & prepare training data ===
df = pd.read_excel('Heat_Map_2025_05_21_to_2025_06_20.xlsx', sheet_name='EnergyConsumption').iloc[1:]
df.columns = ['date', 'hour', 'energy']
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str))
df['energy'] = pd.to_numeric(df['energy'], errors='coerce').fillna(0)
df = df[df['datetime'] < '2025-06-19']

# Feature engineering
df['hour_num'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_sunday'] = (df['day_of_week'] == 6).astype(int)
df['is_monday_morning'] = ((df['day_of_week'] == 0) & (df['hour_num'] < 7)).astype(int)

hour_dummies = pd.get_dummies(df['hour_num'], prefix='hour')
dow_dummies = pd.get_dummies(df['day_of_week'], prefix='dow')

df_prophet = pd.concat([
    df[['datetime', 'energy', 'is_weekend', 'is_sunday', 'is_monday_morning']],
    hour_dummies, dow_dummies
], axis=1)
df_prophet.rename(columns={'datetime': 'ds', 'energy': 'y'}, inplace=True)

# Model
model = Prophet(daily_seasonality=False, weekly_seasonality=False, changepoint_prior_scale=0.3)
model.add_regressor('is_weekend')
model.add_regressor('is_sunday')
model.add_regressor('is_monday_morning')
for col in hour_dummies.columns:
    model.add_regressor(col)
for col in dow_dummies.columns:
    model.add_regressor(col)
model.fit(df_prophet)

# Forecast
future = model.make_future_dataframe(periods=6 * 24, freq='H')
future['hour_num'] = future['ds'].dt.hour
future['day_of_week'] = future['ds'].dt.dayofweek
future['is_weekend'] = future['day_of_week'].isin([5, 6]).astype(int)
future['is_sunday'] = (future['day_of_week'] == 6).astype(int)
future['is_monday_morning'] = ((future['day_of_week'] == 0) & (future['hour_num'] < 7)).astype(int)

hour_dummies_future = pd.get_dummies(future['hour_num'], prefix='hour')
dow_dummies_future = pd.get_dummies(future['day_of_week'], prefix='dow')

# Align columns
for col in hour_dummies.columns:
    hour_dummies_future[col] = hour_dummies_future.get(col, 0)
hour_dummies_future = hour_dummies_future[hour_dummies.columns]

for col in dow_dummies.columns:
    dow_dummies_future[col] = dow_dummies_future.get(col, 0)
dow_dummies_future = dow_dummies_future[dow_dummies.columns]

future = pd.concat([
    future,
    hour_dummies_future,
    dow_dummies_future,
    future[['is_weekend', 'is_sunday', 'is_monday_morning']]
], axis=1)

forecast = model.predict(future)

# Filter forecast
forecast_df = forecast[(forecast['ds'] >= '2025-06-19') & (forecast['ds'] < '2025-06-25')].copy()
forecast_df = forecast_df[['ds', 'yhat']]
forecast_df.columns = ['datetime', 'predicted_energy_kVAh']
forecast_df['hour'] = forecast_df['datetime'].dt.strftime('%H:%M')
forecast_df['date'] = forecast_df['datetime'].dt.date

# === Load actual values ===
actual_df = pd.read_excel(r'C:\Users\IITMRP\PowerPrediction\actual_19to24.xlsx')
actual_df['datetime'] = pd.to_datetime(actual_df['datetime'])

# Merge and compute error
merged = pd.merge(forecast_df, actual_df, on='datetime', how='inner')
merged.rename(columns={'actual_energy': 'actual_energy_kVAh'}, inplace=True)
merged['abs_error'] = abs(merged['predicted_energy_kVAh'] - merged['actual_energy_kVAh'])
merged['abs_perc_error'] = (merged['abs_error'] / merged['actual_energy_kVAh']) * 100

# Metrics
mae = merged['abs_error'].mean()
mape = merged['abs_perc_error'].mean()

# Plot
plt.figure(figsize=(15, 7))
for date in merged['date'].unique():
    subset = merged[merged['date'] == date]
    plt.plot(subset['hour'], subset['predicted_energy_kVAh'], marker='o', label=str(date))
    for x, y, err in zip(subset['hour'], subset['predicted_energy_kVAh'], subset['abs_perc_error']):
        plt.text(x, y + 1, f"{err:.1f}%", ha='center', fontsize=8, color='crimson')

plt.title("Forecasted vs Actual Hourly Energy (June 19–24)\nInaccuracy Shown Above Each Point", fontsize=14)
plt.xlabel("Hour of Day")
plt.ylabel("Predicted Energy (kVAh)")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Date", loc='upper right')

# Annotate MAE + MAPE
plt.text(0.01, 0.95, f"MAE = {mae:.2f} kVAh\nMAPE = {mape:.2f}%", 
         transform=plt.gca().transAxes, fontsize=12, color='black',
         bbox=dict(facecolor='lightyellow', edgecolor='gray', boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.show()
