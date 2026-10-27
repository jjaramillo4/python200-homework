import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from weather_model import WeatherClassifier, Prediction

FEATURES = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]

GOOD_DAY = {"temperature_2m_max": 22, "temperature_2m_min": 12, "precipitation_sum": 0, "wind_speed_10m_max": 8}
BAD_DAY  = {"temperature_2m_max": 2,  "temperature_2m_min": -5, "precipitation_sum": 25, "wind_speed_10m_max": 50}


@pytest.fixture
def classifier(tmp_path):
    train = pd.DataFrame([
        GOOD_DAY,
        {"temperature_2m_max": 19, "temperature_2m_min": 9, "precipitation_sum": 0.5, "wind_speed_10m_max": 12},
        {"temperature_2m_max": 25, "temperature_2m_min": 15, "precipitation_sum": 0, "wind_speed_10m_max": 5},
        BAD_DAY,
        {"temperature_2m_max": 5, "temperature_2m_min": -2, "precipitation_sum": 18, "wind_speed_10m_max": 40},
        {"temperature_2m_max": 0, "temperature_2m_min": -8, "precipitation_sum": 30, "wind_speed_10m_max": 55},
    ])
    labels = [1, 1, 1, 0, 0, 0]
    model = LogisticRegression().fit(train[FEATURES], labels)
    path = tmp_path / "model.pkl"
    joblib.dump(model, path)
    return WeatherClassifier(path)

@pytest.mark.parametrize("day, expected_label", [
    (GOOD_DAY, "good"),
    (BAD_DAY, "skip"),
])
def test_good_label(classifier, day, expected_label):

def test_empty_list_returns_empty(classifier):

def test_probability_in_range:

def test_three_days_in_order:

def test_missing_features_raises:

def test_missing_file_raises: