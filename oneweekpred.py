import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LinearRegression

# === Load cleaned data ===
df = pd.read_excel("cleaned_8thto1st.xlsx")
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.sort_values('DateTime')
df['hour'] = df['DateTime'].dt.hour
df['weekday'] = df['DateTime'].dt.day_name()

# === Feature Engineering ===
X = pd.get_dummies(df[['hour', 'weekday']], drop_first=False)
y = df['Energy_kVAh'].values

# === Train model ===
model = LinearRegression()
model.fit(X, y)

# === Create next 7 days hourly timestamps ===
last_time = df['DateTime'].iloc[-1]
future_times = [last_time + timedelta(hours=i + 1) for i in range(168)]
future_df = pd.DataFrame({'DateTime': future_times})
future_df['hour'] = future_df['DateTime'].dt.hour
future_df['weekday'] = future_df['DateTime'].dt.day_name()
future_df['date'] = future_df['DateTime'].dt.date

# === Predict future values ===
X_future = pd.get_dummies(future_df[['hour', 'weekday']], drop_first=False)
for col in X.columns:
    if col not in X_future:
        X_future[col] = 0
X_future = X_future[X.columns]

future_df['Predicted_kVAh'] = model.predict(X_future)

# === Line Plot: One line per day ===
colors = plt.cm.get_cmap('tab10', 7)  # Up to 7 unique day colors
unique_dates = future_df['date'].unique()

plt.figure(figsize=(15, 6))
for i, date in enumerate(unique_dates):
    subset = future_df[future_df['date'] == date]
    plt.plot(subset['DateTime'], subset['Predicted_kVAh'], label=str(date), color=colors(i), marker='o')

plt.title("Hourly Predicted Energy Usage: Next 7 Days")
plt.xlabel("Datetime")
plt.ylabel("Predicted Energy (kVAh)")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Date", bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()
