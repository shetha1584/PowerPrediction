import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your energy data
df = pd.read_excel('Heat_Map_2025_05_21_to_2025_06_20.xlsx', sheet_name='EnergyConsumption').iloc[1:]
df.columns = ['date', 'hour', 'energy']
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str))
df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
df.dropna(subset=['datetime', 'energy'], inplace=True)

# --- Step 1: Aggregate energy per day ---
df['day'] = df['datetime'].dt.date
daily_energy = df.groupby('day')['energy'].sum()

# --- Step 2: Calculate Z-score ---
mean = daily_energy.mean()
std = daily_energy.std()
z_scores = (daily_energy - mean) / std

# --- Step 3: Detect anomalies (Z > 2 or Z < -2) ---
anomalies = daily_energy[np.abs(z_scores) > 2]

# --- Step 4: Plot ---
plt.figure(figsize=(10, 5))
plt.plot(daily_energy.index, daily_energy, label='Daily Energy', marker='o')
plt.scatter(anomalies.index, anomalies, color='red', label='Anomalies', zorder=5)
plt.axhline(mean, color='green', linestyle='--', label='Mean')
plt.title('Daily Energy Consumption with Anomalies')
plt.xlabel('Date')
plt.ylabel('Total Energy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Optional: View anomaly dates ---
print("Anomaly Dates:")
print(anomalies)
