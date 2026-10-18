import numpy as np
import pandas as pd
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

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y
 )
#--- Preprocessing ---

# Question 1

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.mean(axis=0))

# Because the scaler on X_train only prevents data leakage from the test into the training process. 

# Question 2
encoder = OneHotEncoder(sparse_output=False)

season_col = encoder.fit_transform(df[["season"]])

print(season_col.shape)

print(encoder.get_feature_names_out())

# Season turned into 4 columns because there are 4 distinct categories eaching getting its own binary indicator column. 

# Question 3

X_all = df[numeric + ["season"]]

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
   transformers = [
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(sparse_output=False),
    ["season"]),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

pipeline.fit(X_train_all, y_train_all)

print(pipeline.score(X_test_all, y_test_all))

# --- KNN ---

# Question 1

# Question 2

# Question 3

# --- Logistic Regression ---

# Question 1

# Question 2

# Question 3

# --- Evaluation ---

# Question 1

# --- joblib ---

# Question 1