from pathlib import Path
import joblib
import json
from dataclasses import dataclass

# Warmup Question 1

class ModelInfo:
    def __init__(self,path):
        model_path = Path(path)
        with open(model_path) as f:
            self.data = json.load(f)

    def feature_count(self) -> int:
       return len(self.data["features"])
    
sample = {"features": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]}
with open("model_info_sample.json", "w") as f:
    json.dump(sample, f)


info = ModelInfo("model_info_sample.json")
print(info.feature_count())

#__init__ reads the file once when the object is created. feature_count() would be slow since it reads on every call.

# Warmup Question 2

@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool
    
    def summary(self) -> str:
        if self.will_rain:
         rain_text = "Rain anticipated"
        else:
         rain_text = "Rain not likely"

        return f"{self.date} High: {self.high_c}C, {rain_text}" 
    
day1 = DayForecast("2023-10-15", 24.1, True)
day2 = DayForecast("2023-10-16", 22.3, False)
print(day1.summary())
print(day2.summary())

# I get __init__ and __repr__ without creating it. 

# Warmup Question 3
