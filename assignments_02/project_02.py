import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 33.94,
    "longitude": -84.21,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

# Task 1

# City is selected is Norcross
print(df.shape)
print(df.head(5))

# Task 2
print(df.describe())
print(df.isna().sum())
# No missing values in the dataset, so no need to handle missing data.
print(df.corr(numeric_only=True)["temperature_2m_max"].sort_values())
#Temperature_2m_min is the most correlated with temperature_2m_max, which makes sense since they are both temperatures. 

plt.figure()
plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"])
plt.title("Temperature Correlation")
plt.xlabel("Minimum Temperature")
plt.ylabel("Maximum Temperature")
plt.savefig("outputs/task_2_correlation.png")
plt.show()

plt.figure()
plt.plot(df["date"], df["temperature_2m_max"])
plt.title("Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Maximum Temperature")
plt.savefig("outputs/task_2_trend.png")
plt.show()

# Based on the correlation and scatter plot, it seems that there is a positive linear relationship between temperature_2m_min and temperature_2m_max.
# As the minimum temperature increases, the maximum temperature also tends to increase. 
# This suggests that we can use temperature_2m_min as a feature to predict temperature_2m_max using linear regression.
# Also the trend plot shows that the maximum temperature varies over time, with some seasonal patterns specifically in the summer months.


# Task 3
df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)

print(df["is_summer"].value_counts())

# I think is_summer is a good feature to include in the model because it captures the seasonal effect on temperature.

# Task 4
shaped_2m_min = df["temperature_2m_min"].values.reshape(-1,1)
stored_2m_max = df["temperature_2m_max"]
model = LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(shaped_2m_min, stored_2m_max, test_size=0.2, random_state=42)

model.fit(x_train, y_train)

predicted_high = model.predict(x_test)
print("Slope:", model.coef_[0])
print("Model intercept:", model.intercept_)
print("RMSE:", np.sqrt(mean_squared_error(y_test, predicted_high)))
print("R-squared:", r2_score(y_test, predicted_high))

#RMSE is 3.1 which means that the model's predictions are off about 3.1 celisius on average. That means the model could be off by 5-6 degrees Fahrenheit.

# Task 5
full_model = LinearRegression()
feature_cols = ["temperature_2m_min", "precipitation_sum",
                "wind_speed_10m_max", "is_summer"]
X = df[feature_cols].values
y = df["temperature_2m_max"].values

full_x_train, full_x_test, full_y_train, full_y_test = train_test_split(X, y, test_size=0.2, random_state=42)
full_model.fit(full_x_train, full_y_train)
full_predicted_high = full_model.predict(full_x_test)
for name, coef in zip(feature_cols, full_model.coef_):
    print(f"{name:22s}: {coef:+.3f}")

print("Train R-squared:", full_model.score(full_x_train, full_y_train))
print("Test R-squared:", r2_score(full_y_test, full_predicted_high))
print("Test RMSE:", np.sqrt(mean_squared_error(full_y_test, full_predicted_high)))

#Precipitation_sum has a negative coefficient which means that as precipitation increases, the maximum temperature decreases. 
# This makes sense because rainy days are usually cooler than dry days.
# is_summer has a positive coefficient which means that during the summer months, the maximum temperature is expected to be higher than in other months.
# temperature_2m_min has the largest positive coefficient which means that it has the strongest effect on the maximum temperature.
# No signs were surprising in the data.
# Test R² went from 0.864 (baseline) to 0.902 (full model), so adding features helped.
# Train R² (0.878) and test R² (0.902) are close, so the model is not overfitting.


# Task 6

plt.figure()
plt.scatter(full_predicted_high, full_y_test)
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted High")
plt.ylabel("Actual High")
plt.plot([full_y_test.min(), full_y_test.max()], [full_y_test.min(), full_y_test.max()], linestyle="--") 
plt.savefig("outputs/predicted_vs_actual_high.png")
plt.show()

#Comments
# The model struggles more at the low end: on cooler days the dots spread out further from the line.
# The size of the dataset and the test set: 365 rows which is days of data with 5 columns. 73 days which is 20% of the data is used for testing. The rest is used for training.
# The RMSE and R² of your full model: RMSE 2.62 (about 4.7°F) and R² 0.902.
# Largest effect: is_summer (+1.879) raises the predicted high the most; summer days are predicted about 1.9 degrees warmer.
# One result that surprised you: All the results were surprising, not used to thinking about how conditions affect weather, but humidity in the summer months in Atlanta. 