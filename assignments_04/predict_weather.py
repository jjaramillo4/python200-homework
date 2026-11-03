from weather_model import WeatherClassifier

# Task 5
DAYS = [
    # clearly good: mild, dry, calm
    {"temperature_2m_max": 21, "temperature_2m_min": 12, "precipitation_sum": 0, "wind_speed_10m_max": 10},
    # clearly bad: freezing, rainy, windy
    {"temperature_2m_max": 3, "temperature_2m_min": -4, "precipitation_sum": 20, "wind_speed_10m_max": 45},
    # borderline: just inside every limit
    {"temperature_2m_max": 25.5, "temperature_2m_min": 0.5, "precipitation_sum": 2.5, "wind_speed_10m_max": 28},
    # borderline: just over the rain limit
    {"temperature_2m_max": 18, "temperature_2m_min": 8, "precipitation_sum": 3.5, "wind_speed_10m_max": 15},
]

def main():
    classifier = WeatherClassifier("models/weather_classifier.pkl")
    prediction = classifier.predict(DAYS)

    for day, pred in zip(DAYS, prediction):
        print(
            f"High {day['temperature_2m_max']}C, low {day['temperature_2m_min']}C, "
            f"rain {day['precipitation_sum']}mm, wind {day['wind_speed_10m_max']}km/h "
            f"-> {pred.label} ({pred.probability:.0%})"
        )


if __name__ == "__main__":
    main()
# The guard prevents main() from running automatically on import. 

# Task 6
# 1) The model only sees numbers by position, not by name. If a caller's dict listed
#    wind before temperature, the model would read 45 km/h wind as a 45C high and give
#    a wrong label without any error. A fixed order prevents numbers from different fields mixing.
# 2) We control the training data, so we know what the answers should be and can compare
#    and match. The tests also don't break if the real .pkl is missing, retrained, or changed.
# 3) Ready: predict() takes a whole batch at once, keeps the order, and raises a clear
#    ValueError on bad records. Before production I'd add logging, so bad records and
#    failed predictions are recorded instead of just crashing the batch.