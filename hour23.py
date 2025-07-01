import pandas as pd
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_excel('Heat_Map_2025_05_21_to_2025_06_20.xlsx', sheet_name='EnergyConsumption').iloc[1:]
df.columns = ['date', 'hour', 'energy']

# Convert to datetime
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str))
df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
df.dropna(subset=['datetime', 'energy'], inplace=True)

# Filter only hour 23 and before June 19
df['hour_num'] = df['datetime'].dt.hour
df_23 = df[(df['hour_num'] == 23) & (df['datetime'] < '2025-06-19')]

# Extract just the date for labeling
df_23['date_only'] = df_23['datetime'].dt.date

# Plot
plt.figure(figsize=(14, 6))
plt.bar(df_23['date_only'].astype(str), df_23['energy'], color='teal')
plt.title('Energy Consumption at 23:00 (May 21 – June 18, 2025)')
plt.xlabel('Date')
plt.ylabel('Energy (kVAh) at 23:00')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
