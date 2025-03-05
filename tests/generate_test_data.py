#!/usr/bin/env python3
"""
Generate test data for C++ ML tools using scikit-learn
"""

import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel

def generate_polyregression_data():
    """Generate test data for polynomial regression"""
    print("Generating polynomial regression test data...")
    
    # Create a simple dataset
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 2.0], [5.0, 5.0]])
    y = 2.0 + 3.0 * X[:, 0] + 4.0 * X[:, 1] + 5.0 * X[:, 0]**2
    
    # Fit a polynomial regression model
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Save the model in the format expected by the C++ code
    with open("polyregression_test_model.txt", "w") as f:
        f.write(f"features: {X.shape[1]}\n")
        f.write(f"intercept: {model.intercept_}\n")
        f.write("coefficients:\n")
        coefs_str = ",".join([str(c) for c in model.coef_])
        f.write(f"{coefs_str}\n")
        f.write("powers:\n")
        
        # Get the powers from the polynomial features
        powers = poly.powers_
        for power in powers:
            power_str = ",".join([str(p) for p in power])
            f.write(f"{power_str}\n")
    
    # Save some test points and expected predictions
    with open("polyregression_test_data.txt", "w") as f:
        test_points = np.array([[1.5, 2.5], [3.5, 1.5], [2.5, 3.5]])
        for point in test_points:
            point_poly = poly.transform([point])[0]
            prediction = model.intercept_ + np.sum(model.coef_ * point_poly)
            point_str = ",".join([str(p) for p in point])
            f.write(f"{point_str},{prediction}\n")
    
    print("Polynomial regression test data saved to polyregression_test_model.txt and polyregression_test_data.txt")

def generate_gp_data():
    """Generate test data for Gaussian Process regression"""
    print("Generating Gaussian Process test data...")
    
    # Create a simple dataset
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 2.0], [5.0, 5.0]])
    y = np.sin(X[:, 0]) * np.cos(X[:, 1])
    
    # Fit a GP model with RBF kernel
    kernel = RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
    gp.fit(X, y)
    
    # Save the model in the format expected by the C++ code
    os.makedirs("gp_test_data", exist_ok=True)
    
    # Save the mean
    with open("gp_test_data/gp_mean.txt", "w") as f:
        f.write(f"features: {X.shape[1]}\n")
        f.write("data:\n")
        for val in gp.y_train_:
            f.write(f"{val}\n")
    
    # Save the covariance (just the diagonal for simplicity)
    with open("gp_test_data/gp_cov.txt", "w") as f:
        f.write(f"features: {X.shape[1]}\n")
        f.write("data:\n")
        f.write(f"{gp.alpha}\n")
    
    # Save the kernel
    with open("gp_test_data/gp_mean.txt.kernel", "w") as f:
        f.write("kernel: rbf\n")
        f.write(f"params: {gp.kernel_.length_scale}\n")
    
    # Save the X training data
    with open("gp_test_data/gp_x.txt", "w") as f:
        f.write(f"features: {X.shape[1]}\n")
        f.write("data:\n")
        for point in X:
            point_str = ",".join([str(p) for p in point])
            f.write(f"{point_str}\n")
    
    # Save some test points and expected predictions
    with open("gp_test_data/gp_test_data.txt", "w") as f:
        test_points = np.array([[1.5, 2.5], [3.5, 1.5], [2.5, 3.5]])
        for point in test_points:
            prediction = gp.predict([point])[0]
            point_str = ",".join([str(p) for p in point])
            f.write(f"{point_str},{prediction}\n")
    
    print("Gaussian Process test data saved to gp_test_data/ directory")

if __name__ == "__main__":
    generate_polyregression_data()
    generate_gp_data()
    print("All test data generation complete!")
