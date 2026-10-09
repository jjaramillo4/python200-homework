#-- scikit-learn API --#
import numpy as np
from sklearn.linear_model import LinearRegression

#Q1
temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

model = LinearRegression()
model.fit(temp_min, temp_max) 

new_lows = np.array([[6],[18]])
predicted_new_max = model.predict(new_lows)

print("Predicted maximum temperatures for new lows:", predicted_new_max)

print("Low of 6 Model coefficients:", model.coef_[0])
print("Model intercept:", model.intercept_)

print("Predicted high for low of 6:", predicted_new_max[0])
print("Predicted intercept for low of 18:", predicted_new_max[1])