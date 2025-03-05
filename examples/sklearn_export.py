import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate sample data (2 features)
X = np.array([[2.0, 3.0], [1.0, 4.0], [3.0, 1.0]])
y = np.array([10.0, 16.0, 8.0])  # Some arbitrary target values

# Polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=True)  # include_bias=True includes x^0 = 1
X_poly = poly.fit_transform(X)

# Train linear regression model
model = LinearRegression()
model.fit(X_poly, y)

# Extract coefficients and intercept
intercept = model.intercept_
coefficients = model.coef_
powers = poly.powers_

# Save model to a file (C++ readable format)
with open("model.txt", "w") as f:
    f.write(f"features: {X.shape[1]}\n")
    f.write(f"intercept: {intercept}\n")
    f.write("coefficients:\n")
    f.write(",".join(map(str, coefficients)) + "\n")
    f.write("powers:\n")
    for row in powers:
        f.write(",".join(map(str, row)) + "\n")

# Print predictions for verification
print("Python Model Predictions:")
print(model.predict(X_poly))
