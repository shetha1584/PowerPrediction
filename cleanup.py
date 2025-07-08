import pandas as pd

# Load the Excel file
file_path = r"C:\Users\IITMRP\PowerPrediction\Heat_Map_2025_04_08_to_2025_04_29.xlsx"
df = pd.read_excel(file_path, sheet_name='EnergyConsumption', skiprows=1)

# Rename columns
df.columns = ['date', 'hour', 'energy']

# Drop missing rows
df.dropna(subset=['date', 'hour', 'energy'], inplace=True)

# Combine date and hour into a datetime
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str))

# Convert energy to numeric
df['energy'] = pd.to_numeric(df['energy'], errors='coerce')

# Sort and interpolate missing energy values
df = df.sort_values('datetime').reset_index(drop=True)
df['energy'] = df['energy'].interpolate(method='linear', limit_direction='both')

# Add derived features
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_sunday'] = (df['dayofweek'] == 6).astype(int)
df['is_monday'] = (df['dayofweek'] == 0).astype(int)

# Save cleaned data (optional)
df.to_excel("Cleaned_20days.xlsx", index=False)

# Preview
print(df.head())
