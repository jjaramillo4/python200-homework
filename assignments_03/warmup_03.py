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

# Because the scaler is trained on X_train only prevents data leakage from the test into the training process. 

# Question 2
encoder = OneHotEncoder(sparse_output=False)

season_col = encoder.fit_transform(df[["season"]])

print(season_col.shape)

print(encoder.get_feature_names_out())

# Season turned into 4 columns because there are 4 distinct categories each getting its own binary indicator column. 

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

# It automatically runs the preprocessing steps (scaling and encoding) and feeds the transformed data directly into the model for both training and testing, ensuring no data leaks and no manual data-wrangling steps are needed.

# --- KNN ---

# Question 1

KNN_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

KNN_pipeline.fit(X_train, y_train)

print(classification_report(y_test,KNN_pipeline.predict(X_test)))
print("Test Accuracy:", KNN_pipeline.score(X_test, y_test))

# Question 2

for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    loop_Pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    scores = cross_val_score(loop_Pipeline, X_train, y_train, cv=5)
    mean_score = scores.mean()
    print(k, mean_score)

#  I would use use 11 since it has the highest mean cross-validation accuracy. 

# Question 3

No_Scale_KNN =  Pipeline([
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

No_Scale_KNN.fit(X_train, y_train)

print("No Scale Accuracy:", No_Scale_KNN.score(X_test, y_test))

Scaled_KNN_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

Scaled_KNN_pipeline.fit(X_train, y_train)

print("Scaled Accuracy:", Scaled_KNN_pipeline.score(X_test, y_test))

# Yes scaling helped. It made the model more accurate. This helps us because features with longer range would drown out smaller features. 

# --- Logistic Regression ---

# Question 1

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))

# Question 2

logreg = model.named_steps["logreg"]
for feature, coef in zip(numeric, model.named_steps["logreg"].coef_[0]):
    print(f"{feature}:{coef:.4f}")

# Wind speed is the feature that pushes most towards skip. I would assume it was precipitation, which is also a skip but wind speed affects it more. 
#Both have negative coefficients. 

# Question 3
probs_good = model.predict_proba(X_test[:5])[:,1].round(2)

print(probs_good)

# A Probability of .5 means its a toss up 50/50 that its good to run. 

# --- Evaluation ---

# Question 1

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels =["skip", "good"])
disp.plot()

plt.savefig("outputs/logreg_confusion_matrix.png")
plt.close()

#  how many false positives and how many false negatives did the model make, and which kind of error would matter more for a running app?
# 9 false positives and 7 false negatives. For running, I would assume false positive since heading out to run on a rainly day would be worse.

# --- joblib ---

# Question 1

joblib.dump(model, "models/warmup_model.pkl")

loaded = joblib.load("models/warmup_model.pkl")
assert (loaded.predict(X_test) == model.predict(X_test)).all()
print("Predictions match.")


#  The data would be inaccurate because it would be unscaled. 