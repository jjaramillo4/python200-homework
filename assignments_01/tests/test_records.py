import json
from pathlib import Path

import pytest

from weatherkit import HourlyReading, WeatherResponse, to_readings

DATA_PATH = Path(__file__).parent.parent / "weather_raw.json"


@pytest.fixture
def response():
    return WeatherResponse.model_validate(json.loads(DATA_PATH.read_text()))


def test_one_reading_per_hour_in_order(response):
    readings = to_readings(response)
    assert len(readings) == 168
    assert readings[0].timestamp == "2026-04-08T00:00"
    assert readings[-1].timestamp == "2026-04-14T23:00"


def test_values_line_up_with_timestamps(response):
    readings = to_readings(response)
    hourly = response.hourly
    for i in (0, 50, 167):
        assert readings[i].timestamp == hourly.time[i]
        assert readings[i].temperature_c == hourly.temperature_2m[i]
        assert readings[i].precipitation_mm == hourly.precipitation[i]


def test_identical_readings_are_equal():
    a = HourlyReading("2026-04-08T00:00", 10.0, 0.0)
    b = HourlyReading("2026-04-08T00:00", 10.0, 0.0)
    assert a == b
