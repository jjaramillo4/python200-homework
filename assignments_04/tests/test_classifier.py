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
    result = classifier.predict([day])
    assert len(result) == 1
    assert result[0].label == expected_label

def test_empty_list_returns_empty(classifier):
    assert classifier.predict([]) == []

def test_probability_in_range:
    preds = classifier.predict([GOOD_DAY, BAD_DAY, GOOD_DAY])
    assert 0 <= preds[0].probability <= 1

def test_three_days_in_order:

def test_missing_features_raises:
    bad_day_missing = GOOD_DAY.copy()
    del bad_day_missing["temperature_2m_max"]
    with pytest.raises(FileNotFoundError, match="No such model exists"):
        WeatherClassifier("nonexistent_path.pkl")

def test_missing_file_raises: