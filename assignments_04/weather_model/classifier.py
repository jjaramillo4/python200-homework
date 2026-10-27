from dataclasses import dataclass
from pathlib import Path
import joblib
import pandas as pd

# Task 1
@dataclass
class Prediction:
    """The model's verdict for one day: label is "good" or "skip", and probability is the chance (0 to 1) that the day is good for running."""
    label: str
    probability: float

# Task 2
FEATURES = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]

class WeatherClassifier:
    """Loads a saved weather pipeline once and predicts running weather for many days."""

    def __init__(self, model_path: str | Path) -> None:
        """Load the saved pipeline from model_path.

        Raises:
            FileNotFoundError: if no file exists at model_path.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No such model exists at {model_path}."
            )
        self.model = joblib.load(model_path)
        self.features = FEATURES
    
    def predict(self, days: list[dict]) -> list[Prediction]:
        """Predict whether each day is good for running.

        Raises:
            ValueError: if any day is missing a required feature.
        """
        if not days:
            return []
        frame = pd.DataFrame(days)
        missing = sorted({f for day in days for f in self.features if f not in day})
        if missing:
            raise ValueError(f"missing required features:{missing}")
        
        X = frame[self.features]
        labels = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        return [
            Prediction("good" if int(label) == 1 else "skip", float(prob))
            for label, prob in zip(labels, probabilities)
        ]
