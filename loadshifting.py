import pandas as pd

# Load and clean data
df = pd.read_excel("Heat_Map_2025_05_21_to_2025_06_20.xlsx", sheet_name="EnergyConsumption").iloc[1:]
df.columns = ['date', 'hour', 'energy']

df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str))
df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
df.dropna(subset=['datetime', 'energy'], inplace=True)

# Extract time features
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['weekday'].isin([5, 6])

# Classify Time of Day based on UPPCL ToD tariff
def classify_tod(hour):
    if 5 <= hour < 10:
        return 'morning_discount'
    elif 10 <= hour < 19:
        return 'peak'
    elif 19 <= hour or hour < 3:
        return 'evening_peak'
    else:
        return 'neutral'

df['ToD'] = df['hour'].apply(classify_tod)

# Calculate average usage by hour
weekday_avg = df[~df['is_weekend']].groupby('hour')['energy'].mean()
weekend_avg = df[df['is_weekend']].groupby('hour')['energy'].mean()

# Difference: weekday - weekend
load_shift_diff = (weekday_avg - weekend_avg).sort_values(ascending=False)

# Top 5 hours to shift
top_shift_candidates = load_shift_diff.head(5).reset_index()
top_shift_candidates.columns = ['Hour', 'Weekday - Weekend Energy Difference (kVAh)']

# Merge ToD category for each hour
top_shift_candidates['ToD'] = top_shift_candidates['Hour'].apply(classify_tod)

# Tariff rates based on ToD (in ₹/kWh)
base_tariff = 7.30
tod_multiplier = {
    'morning_discount': 0.85,  # -15%
    'peak': 1.00,
    'evening_peak': 1.15,
    'neutral': 1.00
}

# Compute peak and off-peak rates based on ToD
top_shift_candidates['Weekday Tariff (₹/kWh)'] = top_shift_candidates['ToD'].map(lambda x: base_tariff * tod_multiplier[x])
top_shift_candidates['Weekend Tariff (₹/kWh)'] = base_tariff * 0.85  # weekend assumed to be morning_discount

# Savings calculation
top_shift_candidates['Savings per Hour (₹)'] = (
    top_shift_candidates['Weekday - Weekend Energy Difference (kVAh)'] *
    (top_shift_candidates['Weekday Tariff (₹/kWh)'] - top_shift_candidates['Weekend Tariff (₹/kWh)'])
)

top_shift_candidates['Monthly Savings (₹)'] = top_shift_candidates['Savings per Hour (₹)'] * 20  # assume 20 weekdays

# Display results
total_savings = top_shift_candidates['Monthly Savings (₹)'].sum()
print("Load Shifting Cost Savings Estimate:\n")
print(top_shift_candidates.round(2))
print(f"\nTotal Estimated Monthly Savings: ₹{total_savings:,.2f}")
