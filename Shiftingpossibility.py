import pandas as pd
import matplotlib.pyplot as plt

# Load data
# Load and clean the data
df = pd.read_excel("Heat_Map_2025_05_21_to_2025_06_20.xlsx", sheet_name="EnergyConsumption").iloc[1:]
df.columns = ['date', 'hour', 'energy']

# Combine date and hour into datetime
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str))
df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
df.dropna(subset=['datetime', 'energy'], inplace=True)

# Extract hour and weekday info
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['weekday'].isin([5, 6])

# Calculate hourly average energy usage for weekdays and weekends
weekday_avg = df[~df['is_weekend']].groupby('hour')['energy'].mean()
weekend_avg = df[df['is_weekend']].groupby('hour')['energy'].mean()

# Difference: weekday - weekend usage
load_shift_diff = (weekday_avg - weekend_avg).sort_values(ascending=False)

# Get top 5 hours for potential load shifting
top_shift_candidates = load_shift_diff.head(5).reset_index()
top_shift_candidates.columns = ['Hour', 'Weekday - Weekend Energy Difference (kVAh)']

# Display results
print("Top 5 Load Shifting Candidates (High Weekday Usage vs Weekend):\n")
print(top_shift_candidates)
plt.figure(figsize=(10, 5))
top_shift_candidates.set_index('Hour')['Weekday - Weekend Energy Difference (kVAh)'].plot(
    kind='bar', color='tomato')
plt.title('Top Load Shift Opportunities by Hour')
plt.ylabel('Extra Weekday Usage (kVAh)')
plt.xlabel('Hour of Day')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()