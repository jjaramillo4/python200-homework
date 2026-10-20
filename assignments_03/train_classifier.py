import numpy as np
import pandas as pd
import requests
import json
import platform
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report,
)
import joblib

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 33.94,
    "longitude": -84.21,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

# Task 1

# City is selected is Norcross
print(df.shape)
print(df.head(5))

# Task 2

df["good_for_running"] = ((df["temperature_2m_max"].between(7,26)) & (df["temperature_2m_min"]>=0) & (df["precipitation_sum"]<3) & (df["wind_speed_10m_max"]<30)).astype(int)
print(df["good_for_running"].value_counts())
print("Fractions: ", df["good_for_running"].value_counts(normalize=True))

# 136 days which is about 37% for good for running. This makes sense because winters get below 7 and summers get well above 26, We do have a good amount of rain as well. 
# The data has a mild imbalance. Accuracy helps but we might need f1 scores and precision for good days. 

# Task 3
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y
 )

best_score = 0
best_k = 0

for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=k))
])
    scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_score = scores.mean()
    print(f"K: {k}  Mean CV score: {mean_score:.4f}")

    if(mean_score >= best_score):
        best_score = mean_score
        best_k = k
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])
model.fit(X_train, y_train)

print(classification_report(y_test,model.predict(X_test)))
print("Test Accuracy:", model.score(X_test, y_test))

#Using K=3 since the mean score is .91788

# Task 4

new_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
new_model.fit(X_train, y_train)

y_pred = new_model.predict(X_test)
print(classification_report(y_test, y_pred))

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))

logreg = new_model.named_steps["logreg"]
for feature, coef in zip(numeric, new_model.named_steps["logreg"].coef_[0]):
    print(f"{feature}:{coef:.4f}")

# temperature_2m_max and precipitation_sum drive whether to skip with coefficients of -1.33 and -2.6 respectively. 

# Task 5
# The KNN model is more accurate with a result of .918 and the logistic regression giving us .808
# I think false positives are most important, the worse would be a false positive where users goes out to run and its raining etc.
# Accuracy matters, but precision is more important in this instance since it measures how often a "good" prediction is actually good.
# I would choose the KNN model, since its the most accurate and has higher precision (.82 vs .76). On the logreg model you're looking at a recall of 70% and 76% for precision.
# KNN stores all training days and is slower to predict, but with only ~290 days that is not a problem. The trade-off is that logreg is easier to explain with its coefficients and gives smoother probabilities, while KNN with k=3 can only give 0, .33, .67 or 1.

# Task 6
joblib.dump(model, "models/weather_classifier.pkl")

loaded = joblib.load("models/weather_classifier.pkl")
assert (loaded.predict(X_test) == model.predict(X_test)).all()
print("Predictions match.")
print("File saved")

metadata = {
"python_version": platform.python_version(),
"sklearn_version": sklearn.__version__,
"city": "Norcross, GA",
"features": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
"label_rule": "good if 7 <= high <= 26, low >= 0, precipitation < 3 mm, wind < 30 km/h",
"test_accuracy": float(accuracy_score(y_test, model.predict(X_test))),
"f1_score": float(f1_score(y_test, model.predict(X_test)))
}

with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
    print("JSON doc created: ")

