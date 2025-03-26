import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import math

# --- 1. Load JSON Files ---
dim_calendar = pd.read_json('Hópverkefni4/dimCalendarResults.json')
dim_products = pd.read_json('Hópverkefni4/dimProductsResults.json')
dim_store = pd.read_json('Hópverkefni4/dimStoreResults.json')
fact_inventory = pd.read_json('Hópverkefni4/factInventoryResults.json')
fact_sales = pd.read_json('Hópverkefni4/factSalesResults.json')

# --- 2. Merge Sales with Calendar ---
sales = fact_sales.merge(dim_calendar, left_on='idCalendar', right_on='id', suffixes=('_sales', '_calendar'))
sales['date'] = pd.to_datetime(sales['date'])

# --- 3. Aggregate Sales by Date ---
daily_sales = sales.groupby('date')['unitsSold'].sum().reset_index().sort_values('date')
print("Daily Sales Data (head):")
print(daily_sales.head())

# --- 3a. Log Transform the Sales ---
daily_sales['unitsSoldLog'] = np.log1p(daily_sales['unitsSold'])  
# log1p(x) = log(1 + x), safer if x=0.

# --- 4. Define Training Period with Extra Data for Lags ---
max_date_overall = daily_sales['date'].max()
training_target_start = max_date_overall - pd.Timedelta(days=364)  # last 365 days
lag_start_date = training_target_start - pd.Timedelta(days=365)    # extra year for 365-lag

training_data = daily_sales[daily_sales['date'] >= lag_start_date].copy()
print("\nTraining data range from:", training_data['date'].min(), "to", training_data['date'].max())

# --- 5. Feature Engineering ---
# (A) 365 Lag Features on the *log-transformed* sales
for lag in range(1, 366):
    training_data[f'lag_{lag}'] = training_data['unitsSoldLog'].shift(lag)

# (B) Add Day-of-Year cyclical features
training_data['day_of_year'] = training_data['date'].dt.day_of_year
training_data['day_of_year_sin'] = np.sin(2 * np.pi * training_data['day_of_year'] / 365)
training_data['day_of_year_cos'] = np.cos(2 * np.pi * training_data['day_of_year'] / 365)

# Keep only the target 365 days, dropping rows with missing lags.
training_data = training_data[training_data['date'] >= training_target_start].dropna().reset_index(drop=True)
print("\nTraining Data with 365 Lag Features (log-transformed) and Seasonality (head):")
print(training_data.head())

# --- 6. Train the Random Forest on the log-transformed target ---
lag_features = [f'lag_{lag}' for lag in range(1, 366)]
seasonality_features = ['day_of_year_sin', 'day_of_year_cos']
feature_cols = lag_features + seasonality_features

X_train = training_data[feature_cols]
y_train = training_data['unitsSoldLog']  # Train on log values

# Hyperparameters can be tuned further
model = RandomForestRegressor(n_estimators=200, 
                              max_depth=None, 
                              min_samples_leaf=1, 
                              random_state=42)
model.fit(X_train, y_train)

# --- 7. Forecast the Next 365 Days ---
forecast_horizon = 365

# We'll need the last 365 days of *log-transformed* sales as the seed
last_known_logs = daily_sales.tail(365)['unitsSoldLog'].tolist()

forecast_start_date = daily_sales['date'].max() + pd.Timedelta(days=1)
future_dates = pd.date_range(forecast_start_date, periods=forecast_horizon)

forecast_logs = []  # store predictions in log-space

for i in range(forecast_horizon):
    # (A) Lag features from the last 365 log-sales
    X_lags = np.array(last_known_logs[-365:]).reshape(1, -1)

    # (B) Day-of-year cyclical for this forecast date
    day_of_year = future_dates[i].day_of_year
    day_of_year_sin = math.sin(2 * math.pi * day_of_year / 365)
    day_of_year_cos = math.cos(2 * math.pi * day_of_year / 365)

    X_season = np.array([[day_of_year_sin, day_of_year_cos]])
    X_forecast = np.concatenate([X_lags, X_season], axis=1)

    # (C) Predict in log-space
    y_pred_log = model.predict(X_forecast)[0]
    forecast_logs.append(y_pred_log)

    # (D) Append new log-sales to the seed
    last_known_logs.append(y_pred_log)

# Convert log predictions back to original units
forecast_values = np.expm1(forecast_logs)  # inverse of log1p

forecast_df = pd.DataFrame({
    'date': future_dates,
    'forecast_unitsSold': forecast_values
})

print("\nForecast for the Next Full Year (365 Days):")
print(forecast_df.head())

# --- 8. Plot the Results ---
plt.figure(figsize=(10, 5))
plt.plot(daily_sales['date'], daily_sales['unitsSold'], label='Historical Sales')
plt.plot(forecast_df['date'], forecast_df['forecast_unitsSold'], 
         label='Forecasted Sales (Log + Seasonal)', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.title('Sales Forecast for the Next Full Year (Log Transform + Seasonal Features)')
plt.legend()
plt.show()
