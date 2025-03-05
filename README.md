# C++ Utilities

A collection of modern C++ utilities focused on performance and thread safety.

## Available Utilities

### 1. ConfigLoader
A thread-safe configuration loader with type validation and error handling.

### 2. PolynomialRegression
A thread-safe polynomial regression implementation compatible with scikit-learn models.

---

# ConfigLoader

A modern C++ configuration loader with type safety and validation support. This library provides a flexible and easy-to-use interface for loading and managing configuration values from text files.

## Features

### Key Features

- **Type-Safe Access**: Strongly typed configuration values with compile-time type checking
- **Automatic Type Detection**: Automatically detects and converts values to appropriate types
- **Validation Support**: Built-in validation rules for numbers and strings
- **Flexible Value Types**: Supports:
  - Integers
  - Floating-point numbers
  - Strings
  - Arrays (comma-separated values)
- **Error Handling**: Comprehensive error reporting with line numbers
- **Comments Support**: Ignores comment lines starting with '#'

---

# GP-Predict

A C++ implementation of Gaussian Process regression that supports various kernels and is compatible with scikit-learn models. Features thread-safe prediction and composite kernel support.

## Key Features

- **Multiple Kernels**: 
  - RBF (Radial Basis Function)
  - Matérn (with configurable ν)
  - Constant
  - Composite kernels (sum and product)
- **Thread Safety**: Support for concurrent predictions
- **scikit-learn Compatibility**: Load models trained with scikit-learn's GaussianProcessRegressor
- **High Performance**: Efficient matrix operations using Eigen
- **Type Safety**: Strong type checking and error handling

## Usage

### Training and Exporting from Python

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

# Generate sample data
X = np.linspace(0, 10, 100)[:, np.newaxis]
y = np.sin(X.ravel()) + np.random.normal(0, 0.1, X.shape[0])

# Create composite kernel (Constant + Matern)
kernel = ConstantKernel(1.0) + Matern(length_scale=1.0, nu=1.5)
model = GaussianProcessRegressor(kernel=kernel)
model.fit(X, y)

# Export model (using provided script)
python sklearn_export.py --model gaussian --kernel constant_matern
```

### Basic Usage in C++

```cpp
#include <iostream>
#include "gp-predict.h"
#include <Eigen/Dense>

int main() {
    try {
        GaussianProcess gp;
        
        // Load model with composite kernel
        if (!gp.loadModel("data/mean.txt", "data/covariance.txt")) {
            return -1;
        }

        // Make predictions
        Eigen::VectorXd x(1);
        x << 2.5;
        double prediction = gp.predict(x);
        std::cout << "Prediction at x=2.5: " << prediction << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
```

### Thread-Safe Usage

```cpp
#include <thread>
#include <vector>
#include "gp-predict.h"

void predictConcurrently(const GaussianProcess& gp) {
    std::vector<std::thread> threads;
    std::vector<double> points = {2.0, 4.0, 6.0, 8.0};
    
    for (double x : points) {
        threads.emplace_back([&gp, x]() {
            Eigen::VectorXd input(1);
            input << x;
            double pred = gp.predict(input);
            std::cout << "f(" << x << ") = " << pred << std::endl;
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}
```

### Available Kernels

1. **RBF Kernel**:
   ```python
   kernel = RBF(length_scale=1.0)
   ```

2. **Matérn Kernel**:
   ```python
   kernel = Matern(length_scale=1.0, nu=1.5)  # nu can be 0.5, 1.5, or 2.5
   ```

3. **Constant Kernel**:
   ```python
   kernel = ConstantKernel(constant_value=1.0)
   ```

4. **Composite Kernels**:
   ```python
   # Sum of kernels
   kernel = ConstantKernel(1.0) + Matern(1.0, 1.5)
   
   # Product of kernels
   kernel = ConstantKernel(1.0) * RBF(1.0)
   ```

# PolynomialRegression

A C++ implementation of polynomial regression that can load and use models trained with scikit-learn. Designed for high-performance prediction with thread safety.

## Key Features

- **Scikit-learn Compatibility**: Load models trained with scikit-learn's PolynomialFeatures
- **Thread Safety**: Support for concurrent predictions and model updates
- **High Performance**: Efficient matrix operations using Eigen
- **Batch Processing**: Support for both single and batch predictions
- **Error Handling**: Comprehensive error checking and reporting

## Usage

### Scikit-learn Integration Example

This example demonstrates how to train a model using scikit-learn and use it in C++:

1. **Train Model in Python** (examples/sklearn_export.py):
```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate sample data (2 features)
X = np.array([[2.0, 3.0], [1.0, 4.0], [3.0, 1.0]])
y = np.array([10.0, 16.0, 8.0])  # Some arbitrary target values

# Polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)

# Train and save model
model = LinearRegression()
model.fit(X_poly, y)

# Save model parameters
with open("model.txt", "w") as f:
    f.write(f"features: {X.shape[1]}\n")
    f.write(f"intercept: {model.intercept_}\n")
    f.write("coefficients:\n")
    f.write(",".join(map(str, model.coef_)) + "\n")
    f.write("powers:\n")
    for row in poly.powers_:
        f.write(",".join(map(str, row)) + "\n")
```

2. **Use Model in C++** (examples/model_usage.cpp):
```cpp
#include <iostream>
#include <Eigen/Dense>
#include "polyregression.h"

int main() {
    try {
        PolynomialRegression model;
        model.loadModel("model.txt");  // Load trained sklearn model

        // Example inputs (matching those in Python)
        Eigen::MatrixXd X(3, 2);
        X << 2.0, 3.0,
             1.0, 4.0,
             3.0, 1.0;

        // Make predictions
        std::cout << "Predictions:\n" << model.predictBatch(X) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

### Basic Example

```cpp
#include "config.h"
#include <iostream>

int main() {
    try {
        ConfigLoader config;
        config.load("server.conf");

        // Access values with type safety
        int port = config.get<int>("port");
        std::string host = config.getOr<std::string>("host", "localhost");
        auto allowed_ips = config.get<std::vector<std::string>>("allowed_ips");

        std::cout << "Server configuration loaded:\n"
                  << "Port: " << port << "\n"
                  << "Host: " << host << "\n"
                  << "Allowed IPs: " << allowed_ips.size() << " entries\n";
    } catch (const ConfigError& e) {
        std::cerr << "Configuration error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

### Configuration File Format

```ini
# Server Configuration
port 8080
host localhost
allowed_ips 192.168.1.1,192.168.1.2,192.168.1.3
max_connections 100
timeout 30.5
```

### Adding Validation Rules

```cpp
ConfigLoader config;

// Validate port number range
config.addRangeRule<int>("port", 1024, 65535);

// Validate email format
config.addPatternRule("email", "[\\w.-]+@[\\w.-]+\\.[a-zA-Z]{2,}");

// Custom validation rule
config.addValidationRule(
    "max_connections",
    [](const ConfigLoader::Value& v) {
        try {
            int val = std::get<int>(v);
            return val > 0 && val <= 1000;
        } catch (const std::bad_variant_access&) {
            return false;
        }
    },
    "max_connections must be between 1 and 1000"
);
```

## API Reference

### Core Methods

- `bool load(const std::string& filename)`: Load configuration from a file
- `T get<T>(const std::string& key)`: Get a value with type checking
- `T getOr<T>(const std::string& key, const T& defaultValue)`: Get a value with default
- `bool hasKey(const std::string& key)`: Check if a key exists

### Validation Methods

- `void addRangeRule<T>(const std::string& key, T min, T max)`: Add numeric range validation
- `void addPatternRule(const std::string& key, const std::string& pattern)`: Add regex pattern validation
- `void addValidationRule(const std::string& key, ValidationCallback callback, const std::string& errorMessage)`: Add custom validation

## Error Handling

The library uses the `ConfigError` exception class which provides:
- Detailed error messages
- Line numbers for parsing errors
- Type mismatch information
- Validation failure details

## Building and Installation

This project uses CMake for building and supports selective component compilation. The following components are available:

- **ConfigLoader**: Thread-safe configuration management (header-only)
- **PolynomialRegression**: scikit-learn compatible polynomial regression
- **GP-Predict**: Gaussian Process prediction with composite kernels

### Prerequisites

- CMake 3.15 or higher
- C++17 compliant compiler
- Eigen 3.3 or higher

### Build Options

```cmake
BUILD_POLYREGRESSION  # Build polynomial regression component (ON)
BUILD_GP_PREDICT      # Build Gaussian process component (ON)
BUILD_CONFIG          # Build configuration utility (ON)
BUILD_EXAMPLES        # Build example programs (ON)
```

### Basic Build (All Components)

```bash
# Unix-like systems
mkdir build && cd build
cmake ..
cmake --build .

# Windows with MinGW
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
cmake --build .
```

### Selective Component Build

```bash
# Only polynomial regression
cmake .. -DBUILD_GP_PREDICT=OFF -DBUILD_CONFIG=OFF

# Only Gaussian process
cmake .. -DBUILD_POLYREGRESSION=OFF -DBUILD_CONFIG=OFF

# Only config utility
cmake .. -DBUILD_POLYREGRESSION=OFF -DBUILD_GP_PREDICT=OFF
```

### Installation

```bash
cmake --build . --target install
```

This will install:
- Headers to `${CMAKE_INSTALL_INCLUDEDIR}`
- Libraries to `${CMAKE_INSTALL_LIBDIR}`
- CMake configuration to `${CMAKE_INSTALL_LIBDIR}/cmake`
- Examples to `${CMAKE_INSTALL_BINDIR}` (if built)

### Using in Other Projects

```cmake
find_package(cpp_ml_tools REQUIRED)
target_link_libraries(your_target PRIVATE 
    cpp_ml_tools::polyregression
    cpp_ml_tools::gp-predict
    cpp_ml_tools::config
)
```

## Dependencies

### Required
- C++17 or later
- [Eigen](https://eigen.tuxfamily.org/) 3.4 or later
  - Header-only linear algebra library
  - Used for matrix operations and efficient computations

### Optional (for examples)
- Python 3.7 or later
- scikit-learn
  - For training models and exporting to C++ format
  - Install with: `pip install scikit-learn numpy`

### Build System
- CMake 3.15 or later (recommended)
- Any C++17 compliant compiler:
  - GCC 7 or later
  - Clang 5 or later
  - MSVC 2017 or later

### Thread Safety Requirements
- Standard Library support for:
  - `std::shared_mutex` (C++17)
  - `std::shared_ptr`

## License

This project is open source and available under the MIT License.

```
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
```

### Third-Party Dependencies

- **Eigen**: Mozilla Public License 2.0
  - Only required portions are used
  - Header-only, no linking required
  - Commercial use is permitted

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
