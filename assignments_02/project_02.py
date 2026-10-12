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
print(df.corr(numeric_only=True)["temperature_2m_max"].sort_values())

#plt.scatter()
#plt.title("")
#plt.xlabel("")
#plt.ylabel("")
#plt.savefig("outputs/")
#plt.show()

# Task 3

# Task 4

shaped_2m_min = df["temperature_2m_min"].reshape(-1,1)
stored_2m_max = df["temperature_2m_max"]

model = LinearRegression()

model.fit(shaped_2m_min,stored_2m_max)

predicted_high = model.predict()

# Task 5

# Task 6