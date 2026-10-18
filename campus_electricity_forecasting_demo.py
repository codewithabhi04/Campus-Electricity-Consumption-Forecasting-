"""
Campus Electricity Consumption Forecasting — synthetic demonstration.
IMPORTANT: Input data are synthetic, not actual college meter readings.
Install: pip install pandas numpy matplotlib scikit-learn
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA = "campus_electricity_synthetic_hourly.csv"
df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp")
assert df["timestamp"].is_unique, "Duplicate timestamps found; inspect before modelling."
assert df["consumption_kwh"].notna().all(), "Missing values found; inspect before modelling."

daily = df.set_index("timestamp")["consumption_kwh"].resample("D").sum().to_frame("kwh")
daily["dow"] = daily.index.dayofweek
daily["dayofyear"] = daily.index.dayofyear
daily["lag_1"] = daily["kwh"].shift(1)
daily["lag_7"] = daily["kwh"].shift(7)
daily["roll7"] = daily["kwh"].shift(1).rolling(7).mean()
daily["month"] = daily.index.month
daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
model = daily.dropna()
features = ["dow", "dayofyear", "lag_1", "lag_7", "roll7", "month", "is_weekend"]
split = int(len(model) * 0.8)
train, test = model.iloc[:split], model.iloc[split:]

predictions = {
    "Persistence": test["lag_1"].to_numpy(),
    "Seasonal naive (7d)": test["lag_7"].to_numpy(),
    "7-day moving average": test["roll7"].to_numpy(),
}
rf = RandomForestRegressor(n_estimators=250, random_state=11, min_samples_leaf=3, max_features=0.9)
rf.fit(train[features], train["kwh"])
predictions["Random Forest"] = rf.predict(test[features])

y = test["kwh"].to_numpy()
for name, pred in predictions.items():
    mae = mean_absolute_error(y, pred)
    rmse = mean_squared_error(y, pred) ** 0.5
    mape = np.mean(np.abs((y - pred) / np.maximum(np.abs(y), 1e-9))) * 100
    print(f"{name:24s} MAE={mae:.2f} kWh/day RMSE={rmse:.2f} MAPE={mape:.2f}%")

# Recursive 14-day forecast with Random Forest
history = list(daily["kwh"].values)
future_dates = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=14, freq="D")
forecast = []
for date in future_dates:
    dow = date.dayofweek
    row = pd.DataFrame([{
        "dow": dow, "dayofyear": date.dayofyear, "lag_1": history[-1],
        "lag_7": history[-7], "roll7": float(np.mean(history[-7:])),
        "month": date.month, "is_weekend": int(dow >= 5)
    }])
    value = float(rf.predict(row[features])[0])
    forecast.append(value)
    history.append(value)
pd.DataFrame({"date": future_dates, "forecast_kwh_per_day": forecast}).to_csv(
    "next_14_day_forecast_synthetic.csv", index=False
)
print("Saved next_14_day_forecast_synthetic.csv")
