import pandas as pd
import matplotlib.pyplot as plt

# === Load forecast ===
forecast_df = pd.read_excel('Forestcast_28.xlsx')
forecast_df['datetime'] = pd.to_datetime(forecast_df['datetime'])

# === Load actual ===
actual_df = pd.read_excel('Heat_Map_2025_06_28_to_2025_06_28.xlsx', sheet_name='EnergyConsumption').iloc[1:]
actual_df.columns = ['date', 'hour', 'energy']
actual_df['datetime'] = pd.to_datetime(actual_df['date'].astype(str) + ' ' + actual_df['hour'].astype(str))
actual_df.rename(columns={'energy': 'actual_energy_kVAh'}, inplace=True)
actual_df = actual_df[['datetime', 'actual_energy_kVAh']]

# === Merge forecast and actual ===
merged = pd.merge(forecast_df, actual_df, on='datetime', how='inner')

# === Errors ===
merged['abs_error'] = abs(merged['predicted_energy_kVAh'] - merged['actual_energy_kVAh'])
merged['abs_perc_error'] = (merged['abs_error'] / merged['actual_energy_kVAh']) * 100

# === New percentage column ===
merged['percentage'] = 100 - merged['abs_perc_error']

# === Metrics ===
mae = merged['abs_error'].mean()
mape = merged['abs_perc_error'].mean()
accuracy = 100 - mape

print(f"MAE = {mae:.2f} kVAh")
print(f"MAPE = {mape:.2f}%")
print(f"Accuracy = {accuracy:.2f}%")

# === Save results to Excel ===
output_file = r"C:\Users\HP\PowerPrediction\Accuracy_June28.xlsx"
merged.to_excel(output_file, index=False, engine="openpyxl")
print(f"✅ Accuracy report saved to {output_file}")

# === Plot ===
plt.figure(figsize=(15, 6))
plt.plot(merged['datetime'], merged['predicted_energy_kVAh'], marker='o', label='Forecasted')
plt.plot(merged['datetime'], merged['actual_energy_kVAh'], marker='x', label='Actual')

# Annotate each point with percentage accuracy
for x, y, perc in zip(merged['datetime'], merged['predicted_energy_kVAh'], merged['percentage']):
    plt.text(x, y + 1, f"{perc:.1f}%", ha='center', fontsize=8, color='green')

plt.title(f"Forecast vs Actual Energy (June 28, 2025)\n"
          f"MAE={mae:.2f} kVAh | MAPE={mape:.2f}% | Accuracy={accuracy:.2f}%", fontsize=14)
plt.xlabel("Hour")
plt.ylabel("Energy (kVAh)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
