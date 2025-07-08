import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from datetime import timedelta
from sklearn.metrics import mean_absolute_error

# -------------------------------
# Step 1: Load and Clean April Training Data
# -------------------------------
print("📥 Loading training data...")
train_path = r"C:\Users\IITMRP\PowerPrediction\Cleaned_20days.xlsx"
df = pd.read_excel(train_path)

df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
df['hour'] = df['hour'].astype(str).str.zfill(2)
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'] + ':00')

df['energy'] = pd.to_numeric(df['energy'], errors='coerce').interpolate(method='linear', limit_direction='both')
df['dayofweek'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_sunday'] = (df['dayofweek'] == 6).astype(int)
df['is_monday'] = (df['dayofweek'] == 0).astype(int)

# One-hot encode hour
df['hour'] = df['datetime'].dt.hour
hour_dummies = pd.get_dummies(df['hour'], prefix='hour')
df = pd.concat([df, hour_dummies], axis=1)

features = ['dayofweek', 'is_weekend', 'is_sunday', 'is_monday'] + list(hour_dummies.columns)

print("✅ Training data processed.")

# -------------------------------
# Step 2: Train the Model
# -------------------------------
X = df[features]
y = df['energy']
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
print("✅ Model trained.")

# -------------------------------
# Step 3: Predict May 1–7
# -------------------------------
print("🔮 Predicting May 1–7...")
last_datetime = df['datetime'].max()
future_datetimes = pd.date_range(start=last_datetime + timedelta(hours=1), periods=24*7, freq='H')

future_df = pd.DataFrame({'datetime': future_datetimes})
future_df['hour'] = future_df['datetime'].dt.hour
future_df['dayofweek'] = future_df['datetime'].dt.dayofweek
future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
future_df['is_sunday'] = (future_df['dayofweek'] == 6).astype(int)
future_df['is_monday'] = (future_df['dayofweek'] == 0).astype(int)

# Dip flags (optional)
dip_hours = [6, 7, 13]
future_df['is_dip_hour'] = future_df['hour'].isin(dip_hours).astype(int)
future_df['is_dip_day'] = future_df['dayofweek'].isin([0, 6]).astype(int)
future_df['is_dip_combination'] = ((future_df['is_dip_hour'] == 1) & (future_df['is_dip_day'] == 1)).astype(int)

# One-hot encode future hour and align with training
future_hour_dummies = pd.get_dummies(future_df['hour'], prefix='hour')
for col in hour_dummies.columns:
    if col not in future_hour_dummies:
        future_hour_dummies[col] = 0
future_hour_dummies = future_hour_dummies[hour_dummies.columns]
future_df = pd.concat([future_df, future_hour_dummies], axis=1)

future_df['predicted_energy'] = model.predict(future_df[features])
future_df['date'] = future_df['datetime'].dt.date
future_df.to_excel("Predicted_Energy_Next_Week.xlsx", index=False)
print("✅ Predictions saved.")

# -------------------------------
# Step 4: Load and Clean Actual Data
# -------------------------------
print("📥 Loading actual data (April 29–May 6)...")
actual_path = r"C:\Users\IITMRP\PowerPrediction\Heat_Map_2025_04_29_to_2025_05_06.xlsx"
actual_df = pd.read_excel(actual_path, skiprows=1)
actual_df.columns = ['date', 'hour', 'energy']
actual_df.dropna(subset=['date', 'hour'], inplace=True)

actual_df['date'] = pd.to_datetime(actual_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
actual_df['hour'] = actual_df['hour'].astype(str).str.zfill(2)
actual_df['datetime'] = pd.to_datetime(actual_df['date'] + ' ' + actual_df['hour'] + ':00')
actual_df['energy'] = pd.to_numeric(actual_df['energy'], errors='coerce').fillna(0)
actual_df = actual_df.sort_values('datetime').reset_index(drop=True)
print("✅ Actual data cleaned.")

# -------------------------------
# Step 5: Merge and Evaluate
# -------------------------------
print("🔍 Comparing predictions with actuals...")
predicted_df = pd.read_excel("Predicted_Energy_Next_Week.xlsx")
predicted_df['datetime'] = pd.to_datetime(predicted_df['datetime'])

merged_df = pd.merge(actual_df, predicted_df, on='datetime', how='inner')
merged_df['date'] = merged_df['datetime'].dt.date
merged_df['hour'] = merged_df['datetime'].dt.hour
merged_df['percent_error'] = ((merged_df['predicted_energy'] - merged_df['energy']) / np.maximum(merged_df['energy'], 1)) * 100

# --- 5.1: All Days
mae_full = mean_absolute_error(merged_df['energy'], merged_df['predicted_energy'])
mape_full = np.mean(np.abs(merged_df['percent_error']))
print(f"\n📊 MAE (All Days): {mae_full:.2f}")
print(f"📊 MAPE (All Days): {mape_full:.2f}%")

# --- 5.2: Exclude May 1 & 2
exclude_dates = [pd.to_datetime("2025-05-01").date(), pd.to_datetime("2025-05-02").date()]
filtered_df = merged_df[~merged_df['date'].isin(exclude_dates)].copy()
mae_filtered = mean_absolute_error(filtered_df['energy'], filtered_df['predicted_energy'])
mape_filtered = np.mean(np.abs(filtered_df['percent_error']))
print(f"\n📊 MAE (Excluding May 1 & 2): {mae_filtered:.2f}")
print(f"📊 MAPE (Excluding May 1 & 2): {mape_filtered:.2f}%")

# Save results
merged_df.to_excel("Prediction_Comparison_May1to7.xlsx", index=False)
print("✅ Comparison saved.")

# -------------------------------
# Step 6: Heatmap (excluding May 1 & 2)
# -------------------------------
print("📈 Generating heatmap of % error (excluding May 1 & 2)...")
heatmap_df = merged_df[~merged_df['date'].isin(exclude_dates)]
pivot = heatmap_df.pivot_table(values='percent_error', index='hour', columns='date', aggfunc='mean')

plt.figure(figsize=(14, 6))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap='coolwarm', center=0, cbar_kws={'label': '% Error'})
plt.title("🔍 Heatmap of Prediction % Error (Excluding May 1 & 2)")
plt.xlabel("Date")
plt.ylabel("Hour of Day")
plt.tight_layout()
plt.show()
