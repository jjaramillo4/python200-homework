#Write two Pydantic Models:
#HourlyBlock, with time, temperature_2m and precipitation as lists
#WeatherResponse with top-level fields including hourly: HourlyBlock
#load weather_raw.json validate WeatherResponse and print out Weather response.
from pydantic import BaseModel

class HourlyBlock(BaseModel):
    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

class WeatherResponse(BaseModel):
    latitude: float
    longitude: float
    elevation: float
    utc_offset_seconds: int
    timezone_abbreviation: str 
    timezone: str
    hourly: HourlyBlock
    
if __name__ == "__main__":
    import json
    from pathlib import Path
    path = Path(__file__).parent.parent / "weather_raw.json"
    data = json.loads(path.read_text())
    weather = WeatherResponse.model_validate(data)
    print(len(weather.hourly.time))

