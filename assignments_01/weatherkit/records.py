from dataclasses import dataclass

from weatherkit.schemas import WeatherResponse


# Why a dataclass and not a Pydantic model? The boundary is WeatherResponse:
# that's where untrusted JSON from the API enters the program, so that's where
# validation belongs. By the time we build HourlyReading objects, the data has
# already passed validation, and to_readings is our own code creating them from
# trusted values. Validating again would be wasted work. A dataclass is a plain,
# lightweight container for data we already trust.
@dataclass
class HourlyReading:
    """One hour of weather data.

    timestamp is ISO 8601 local time ("2026-04-08T00:00"), temperature_c is in
    degrees Celsius, and precipitation_mm is in millimeters.
    """
    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert the columnar hourly block into one reading per hour.

    Args:
        response: A validated Open-Meteo response.

    Returns:
        One HourlyReading per hour, in the same order as the input lists.
    """
    hourly = response.hourly
    return [
        HourlyReading(timestamp=t, temperature_c=temp, precipitation_mm=precip)
        for t, temp, precip in zip(hourly.time, hourly.temperature_2m, hourly.precipitation)
    ]
