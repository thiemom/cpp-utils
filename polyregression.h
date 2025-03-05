/*
 * Copyright (c) 2025 Thiemo M.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file polyregression.h
 * @brief Thread-safe polynomial regression implementation compatible with scikit-learn
 *
 * This class provides a C++ implementation of polynomial regression that can load
 * and use models trained with scikit-learn's PolynomialFeatures. It is designed
 * for high-performance prediction with thread safety.
 *
 * Features:
 * - Scikit-learn Compatibility: Load models trained with PolynomialFeatures
 * - Thread Safety: Support for concurrent predictions and model updates
 * - High Performance: Efficient matrix operations using Eigen
 * - Batch Processing: Support for both single and batch predictions
 * - Error Handling: Comprehensive error checking and reporting
 *
 * Example usage:
 * @code
 * PolynomialRegression model;
 * model.loadModel("model.txt");  // Load scikit-learn exported model
 * 
 * // Single prediction
 * Eigen::VectorXd input(2);
 * input << 1.0, 2.0;
 * double result = model.predict(input);
 *
 * // Batch prediction
 * Eigen::MatrixXd inputs(3, 2);
 * inputs << 1.0, 2.0,
 *           2.0, 3.0,
 *           3.0, 4.0;
 * Eigen::VectorXd results = model.predictBatch(inputs);
 * @endcode
 *
 * @note This implementation is thread-safe for both model loading and predictions.
 *       Multiple threads can safely make predictions while the model is being updated.
 *
 * @see sklearn_export.py for the corresponding Python export script
 */

#ifndef POLYNOMIAL_REGRESSION_H
#define POLYNOMIAL_REGRESSION_H

#include <Eigen/Dense>
#include <string>
#include <shared_mutex>
#include <memory>

/**
 * @brief A thread-safe polynomial regression model compatible with scikit-learn.
 *
 * This class implements a polynomial regression model that can load and use models
 * trained with scikit-learn's PolynomialFeatures + LinearRegression. It supports
 * concurrent predictions and thread-safe model updates.
 *
 * Example:
 * @code
 *     PolynomialRegression model;
 *     model.loadModel("model.txt");  // Load scikit-learn exported model
 *     
 *     // Single prediction
 *     Eigen::VectorXd x(2);
 *     x << 1.0, 2.0;
 *     double y = model.predict(x);
 *     
 *     // Batch prediction
 *     Eigen::MatrixXd X(3, 2);
 *     X << 1.0, 2.0,
 *          3.0, 4.0,
 *          5.0, 6.0;
 *     Eigen::VectorXd predictions = model.predictBatch(X);
 * @endcode
 */
/**
 * @brief Thread-safe polynomial regression model compatible with scikit-learn
 *
 * This class implements polynomial regression using coefficients and feature powers
 * exported from a scikit-learn model. It supports both single and batch predictions
 * while maintaining thread safety.
 */
class PolynomialRegression {
private:
    /**
     * @brief Internal structure holding the model parameters.
     * This structure ensures atomic updates of all model parameters.
     */
    struct ModelData {
        int features = 0;
        double intercept = 0.0;
        Eigen::VectorXd coefficients;
        Eigen::MatrixXi powers;
    };

    std::shared_ptr<const ModelData> model;
    mutable std::shared_mutex mutex;

    Eigen::VectorXd parseVector(const std::string& line) const;
    Eigen::MatrixXi parseMatrix(std::ifstream& file, int rows, int cols) const;

public:
    /**
     * @brief Constructs an empty polynomial regression model.
     * The model needs to be loaded using loadModel() before making predictions.
     */
    PolynomialRegression() : model(std::make_shared<ModelData>()) {}

    // Thread-safe model loading
    /**
     * @brief Loads a model from a file exported by scikit-learn.
     * @param filename Path to the model file
     * @throws std::runtime_error if file cannot be opened or has invalid format
     * @thread_safety Thread-safe. Can be called while other threads are making predictions.
     */
    void loadModel(const std::string& filename);

    // Thread-safe prediction methods
    /**
     * @brief Makes a prediction for a single input vector.
     * @param x Input vector of features
     * @return Predicted value
     * @throws std::runtime_error if input size doesn't match model features
     * @thread_safety Thread-safe. Multiple threads can call predict concurrently.
     */
    /**
     * @brief Make a prediction for a single input vector
     *
     * @param x Input vector of features
     * @return Predicted value
     * @throws std::runtime_error if model is not loaded or input size mismatch
     * @thread_safety This operation is thread-safe and can be called concurrently
     */
    double predict(const Eigen::VectorXd& x) const;
    /**
     * @brief Makes predictions for multiple input vectors.
     * @param X Matrix where each row is an input vector
     * @return Vector of predictions
     * @throws std::runtime_error if input column count doesn't match model features
     * @thread_safety Thread-safe. Multiple threads can call predictBatch concurrently.
     */
    /**
     * @brief Make predictions for multiple input vectors
     *
     * @param X Matrix where each row is an input vector
     * @return Vector of predictions
     * @throws std::runtime_error if model is not loaded or input size mismatch
     * @thread_safety This operation is thread-safe and can be called concurrently
     */
    Eigen::VectorXd predictBatch(const Eigen::MatrixXd& X) const;

    // Deleted copy operations to prevent shared mutex issues
    PolynomialRegression(const PolynomialRegression&) = delete;
    PolynomialRegression& operator=(const PolynomialRegression&) = delete;

    // Move operations
    PolynomialRegression(PolynomialRegression&&) noexcept = default;
    PolynomialRegression& operator=(PolynomialRegression&&) noexcept = default;
};

#endif