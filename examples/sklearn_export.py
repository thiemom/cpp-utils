#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file sklearn_export.py
@brief Export scikit-learn models for C++ inference
@copyright MIT License

Copyright (c) 2025 Thiemo M.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


This script demonstrates two ways to export scikit-learn models for use with C++:

1. Polynomial Regression:
   - Uses PolynomialFeatures for feature transformation
   - Exports coefficients and feature powers
   - Simple, deterministic predictions
   Example usage:
   ```python
   python sklearn_export.py --model polynomial --degree 2
   ```

2. Gaussian Process Regression:
   - Uses RBF kernel with white noise
   - Exports mean and covariance matrix
   - Provides uncertainty estimates
   Example usage:
   ```python
   python sklearn_export.py --model gaussian
   ```

Both models are thread-safe in their C++ implementations and can be used
for real-time predictions.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import argparse

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

def generate_data(n_train=20, n_test=10):
    """Generate synthetic training and test data."""
    X_train = np.linspace(0, 10, n_train).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + 0.1 * np.random.randn(n_train)
    
    # Test data includes extrapolation points
    X_test = np.linspace(-1, 11, n_test).reshape(-1, 1)
    y_test = np.sin(X_test).ravel() + 0.1 * np.random.randn(n_test)
    
    return X_train, y_train, X_test, y_test

def train_polynomial_regression(X_train, y_train, degree=2):
    """Train and export polynomial regression model."""
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X_train)
    
    # Train model
    model = LinearRegression()
    model.fit(X_poly, y_train)
    
    return model, poly

def create_kernel(kernel_type='rbf'):
    """Create kernel based on type specification"""
    if kernel_type == 'rbf':
        return RBF(length_scale=1.0)
    elif kernel_type == 'matern':
        return Matern(length_scale=1.0, nu=1.5)
    elif kernel_type == 'constant':
        return ConstantKernel(constant_value=1.0)
    elif kernel_type == 'constant_matern':
        return ConstantKernel(constant_value=1.0) + Matern(length_scale=1.0, nu=1.5)
    else:
        raise ValueError(f'Unsupported kernel type: {kernel_type}')

def train_gaussian_process(X_train, y_train, kernel_type='rbf'):
    """Train and export Gaussian process model with specified kernel."""
    kernel = create_kernel(kernel_type)
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    model.fit(X_train, y_train)
    return model
        kernel = RBF(length_scale=1.0)
    elif kernel_type == 'matern':
        kernel = Matern(length_scale=1.0, nu=1.5)
    elif kernel_type == 'constant':
        kernel = ConstantKernel(constant_value=1.0)
    else:
        raise ValueError(f'Unsupported kernel type: {kernel_type}')
        
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    model.fit(X_train, y_train)
    
    return model
    """Train and export Gaussian process model."""
    kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def save_polynomial_model(model, poly, X_train, y_train, X_test, y_test, data_dir):
    """Save polynomial regression model and metadata."""
    # Cross-validation and test metrics
    cv_scores = cross_val_score(model, poly.fit_transform(X_train), y_train, cv=5, scoring='r2')
    y_pred = model.predict(poly.transform(X_test))
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    # Save model parameters
    np.savetxt(data_dir / 'coefficients.txt', model.coef_)
    np.savetxt(data_dir / 'powers.txt', poly.powers_)
    np.savetxt(data_dir / 'test_input.txt', X_test)
    np.savetxt(data_dir / 'test_output.txt', y_test)
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'polynomial_regression',
        'model_info': {
            'degree': poly.degree,
            'n_features': poly.n_features_in_
        },
        'performance': {
            'cv_score_mean': float(cv_scores.mean()),
            'cv_score_std': float(cv_scores.std()),
            'test_mse': float(test_mse),
            'test_r2': float(test_r2)
        }
    }
    
    return metadata

def save_composite_kernel(kernel, filepath):
    """Save a composite kernel to file"""
    if isinstance(kernel, (RBF, Matern, ConstantKernel)):
        return save_kernel_params(kernel, filepath)

    # Handle composite kernels
    with open(filepath, 'w') as f:
        f.write('composite\n')
        
        # Determine operation type
        if isinstance(kernel, Sum):
            f.write('sum\n')
        elif isinstance(kernel, Product):
            f.write('product\n')
        else:
            raise ValueError(f'Unsupported composite type: {type(kernel)}')
        
        # Save sub-kernels
        k1, k2 = kernel.k1, kernel.k2
        save_kernel_params(k1, filepath + '.k1')
        save_kernel_params(k2, filepath + '.k2')

def save_kernel_params(kernel, filepath):
    """Save kernel parameters to file"""
    with open(filepath, 'w') as f:
        if isinstance(kernel, RBF):
            f.write('rbf\n')
            f.write(f'{kernel.length_scale}\n')
        elif isinstance(kernel, Matern):
            f.write('matern\n')
            f.write(f'{kernel.length_scale}\n')
            f.write(f'{kernel.nu}\n')
        elif isinstance(kernel, ConstantKernel):
            f.write('constant\n')
            f.write(f'{kernel.constant_value}\n')
        else:
            raise ValueError(f'Unsupported kernel type: {type(kernel)}')

def save_training_data(X, filepath):
    """Save training points to file"""
    with open(filepath, 'w') as f:
        for x in X:
            f.write(' '.join(map(str, x)) + '\n')

def save_gaussian_process_model(model, X_train, y_train, X_test, y_test, data_dir):
    """Save Gaussian process model and metadata."""
    # Cross-validation and test metrics
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    y_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    # Save model parameters
    mean = model.predict(X_train)
    covariance_matrix = model.kernel_(X_train)
    # Save mean and covariance
    np.savetxt(data_dir / 'mean.txt', mean)
    np.savetxt(data_dir / 'covariance.txt', covariance_matrix)
    
    # Save kernel parameters and training data
    save_kernel_params(model.kernel_, data_dir / 'mean.txt.kernel')
    save_training_data(X_train, data_dir / 'mean.txt.train')
    np.savetxt(data_dir / 'covariance.txt', covariance_matrix)
    np.savetxt(data_dir / 'kernel_params.txt', [model.kernel_.theta])
    np.savetxt(data_dir / 'test_input.txt', X_test)
    np.savetxt(data_dir / 'test_output.txt', y_test)
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'gaussian_process',
        'model_info': {
            'kernel_type': str(model.kernel_),
            'optimizer_restarts': model.n_restarts_optimizer
        },
        'performance': {
            'cv_score_mean': float(cv_scores.mean()),
            'cv_score_std': float(cv_scores.std()),
            'test_mse': float(test_mse),
            'test_r2': float(test_r2)
        },
        'learned_params': {
            'kernel_params': [float(p) for p in model.kernel_.theta],
            'kernel_description': str(model.kernel_)
        }
    }
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description='Export scikit-learn models for C++')
    parser.add_argument('--model', choices=['polynomial', 'gaussian'], required=True,
                        help='Type of model to train')
    parser.add_argument('--kernel', choices=['rbf', 'matern', 'constant', 'constant_matern'], default='rbf',
                        help='Kernel type for Gaussian process')
    parser.add_argument('--degree', type=int, default=2,
                        help='Polynomial degree (only for polynomial regression)')
    args = parser.parse_args()
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Generate data
        X_train, y_train, X_test, y_test = generate_data()
        
        # Train and save appropriate model
        if args.model == 'polynomial':
            model, poly = train_polynomial_regression(X_train, y_train, args.degree)
            metadata = save_polynomial_model(model, poly, X_train, y_train, X_test, y_test, data_dir)
            print(f"Trained polynomial regression model (degree {args.degree})")
        else:
            model = train_gaussian_process(X_train, y_train, args.kernel)
            metadata = save_gaussian_process_model(model, X_train, y_train, X_test, y_test, data_dir)
            print(f"Trained Gaussian process model with kernel: {model.kernel_}")
        
        # Save metadata
        with open(data_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Model and metadata saved in {data_dir}/")
        print(f"Cross-validation R² score: {metadata['performance']['cv_score_mean']:.3f} "
              f"(+/- {metadata['performance']['cv_score_std']:.3f})")
        print(f"Test set R²: {metadata['performance']['test_r2']:.3f}")
        
    except Exception as e:
        print(f"Error during model training or saving: {str(e)}")

if __name__ == '__main__':
    main()
