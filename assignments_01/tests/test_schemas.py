import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit import WeatherResponse

# A plain relative path like "weather_raw.json" is resolved from the directory
# pytest was launched in, not from this file, so it breaks if you run pytest
# from tests/ or the repo root. Building the path from __file__ always points
# at assignments_01/ no matter where pytest starts.
DATA_PATH = Path(__file__).parent.parent / "weather_raw.json"


def make_response(**overrides):
    """A small valid response; pass keyword args to replace parts of it."""
    data = {
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "GMT",
        "elevation": 254.0,
        "hourly": {
            "time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
            "temperature_2m": [10.0, 11.0, 12.0],
            "precipitation": [0.0, 0.2, 0.0],
        },
    }
    data.update(overrides)
    return data


def test_real_file_validates():
    data = json.loads(DATA_PATH.read_text())
    response = WeatherResponse.model_validate(data)
    assert len(response.hourly.time) == 168


@pytest.mark.parametrize("field, value", [
    ("latitude", 200.0),
    ("latitude", -91.0),
    ("longitude", 181.0),
])
def test_out_of_range_coordinates_raise(field, value):
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(make_response(**{field: value}))


def test_mismatched_lengths_raise():
    hourly = {
        "time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
        "temperature_2m": [10.0, 11.0],
        "precipitation": [0.0, 0.2, 0.0],
    }
    with pytest.raises(ValidationError, match="differ in length"):
        WeatherResponse.model_validate(make_response(hourly=hourly))


def test_null_temperature_raises():
    hourly = {
        "time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
        "temperature_2m": [10.0, None, 12.0],
        "precipitation": [0.0, 0.2, 0.0],
    }
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(make_response(hourly=hourly))
