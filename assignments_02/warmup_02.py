#-- scikit-learn API --#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#Q1
temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

model = LinearRegression()
model.fit(temp_min, temp_max) 

new_lows = np.array([[6],[18]])
predicted_new_max = model.predict(new_lows)

print("Predicted maximum temperatures for new lows:", predicted_new_max)

print("Slope:", model.coef_[0])
print("Model intercept:", model.intercept_)

print("Predicted high for low of 6:", predicted_new_max[0])
print("Predicted high for low of 18:", predicted_new_max[1])


#Q2
x = np.array([10,20,30,40,50]) 

print("Original shape:", x.shape)
print("Reshaped shape:", x.reshape(-1, 1).shape)

#Scikit-learn requires a 2D array for the input features, so we reshape the 1D array x into a 2D array with one column using x.reshape(-1, 1). 
# This allows us to use it as input for machine learning models that expect a 2D array since a 1D array is ambiguous in terms of features and samples.

# --- Linear Regression ---

rng = np.random.default_rng(42)
num_days = 120
temp_min = rng.uniform(-5, 22, num_days)
is_rainy = rng.integers(0, 2, num_days).astype(float)
temp_max = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

#Q1
plt.scatter(temp_min, temp_max, c=is_rainy, cmap='coolwarm')
plt.title("Daily High vs Low")
plt.xlabel("Minimum Temperature")
plt.ylabel("Maximum Temperature")
plt.savefig("outputs/high_vs_low.png")
plt.show()

#Is Rainy lets us see the effect of rain on max temperature. If it rained it usually was less warm. Red rainy days sit below blue dry days. It adds additonal informatio so should
# be a feature in the model.

#Q2

#Q3

#Q4

#Q5