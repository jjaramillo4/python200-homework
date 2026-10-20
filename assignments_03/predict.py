import json
import joblib
import pandas as pd

# Task 7
loaded = joblib.load("models/weather_classifier.pkl")

with open("models/weather_classifier_metadata.json", "r") as f:
    data = json.load(f)
print("City:", data["city"])
print("Features:", data["features"])
print("Test accuracy:", round(data["test_accuracy"], 3))

fake_data_df = pd.DataFrame(
    [
        [18.0, 8.0, 0.0, 10.0],    # clearly good: mild, dry, calm
        [34.0, 24.0, 15.0, 35.0],  # clearly bad: hot, rainy, windy
        [25.8, 15.0, 2.8, 20.0],   # borderline: high and rain just under the limits
        [8.0, 0.5, 0.0, 28.0],     # borderline: cold and windy, but inside the limits
    ],
    columns=data["features"],
)

labels = loaded.predict(fake_data_df)
probs = loaded.predict_proba(fake_data_df)[:,1]

for i in range(len(fake_data_df)):
    row = fake_data_df.iloc[i]
    label = "good" if labels[i] == 1 else "skip"
    print(row.to_dict(), label, round(probs[i], 2))
# Task 8
# 1. My borderline day got 0.67, which is uncertain, so for a 0.52 I would show a "borderline" warning instead of "good".
# 2. joblib.load would crash with FileNotFoundError, so I would check the file exists and print "run train_classifier.py first".
# 3. The fake days would be replaced with each day's real weather from the API.

