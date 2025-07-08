import pandas as pd

# Load the Excel file
# Use full path
excel_path = r"C:\Users\IITMRP\PowerPrediction\Heat_Map_2025_04_08_to_2025_05_01.xlsx"
  # Update path if needed
excel_file = pd.ExcelFile(excel_path)

# Load and clean the 'EnergyConsumption' sheet
df = pd.read_excel(excel_file, sheet_name='EnergyConsumption', skiprows=1)
df.columns = ['date', 'hour', 'energy']  # Rename columns

# Drop any rows with missing key values
df.dropna(subset=['date', 'hour', 'energy'], inplace=True)

# Combine date and hour into a datetime column
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str))

# Convert energy to numeric (handles any stray text)
df['energy'] = pd.to_numeric(df['energy'], errors='coerce')

# Sort data and reset index
df = df.sort_values('datetime').reset_index(drop=True)

# Fill missing energy values using linear interpolation
df['energy'] = df['energy'].interpolate(method='linear', limit_direction='both')

# Extract features for modeling
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek  # 0 = Monday, 6 = Sunday
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_sunday'] = (df['dayofweek'] == 6).astype(int)
df['is_monday'] = (df['dayofweek'] == 0).astype(int)

# Preview the cleaned and enriched data
# Save to Excel
df.to_excel("Cleanedweek.xlsx", index=False)

# Sorted list of unique dates
print(sorted(df['date'].unique()))

