/**
 * @file gp-predict.h
 * @brief Thread-safe Gaussian Process prediction from scikit-learn models
 * @copyright MIT License
 *
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
 *
 * @details
 * This class provides a thread-safe implementation for making predictions using
 * Gaussian Process models trained with scikit-learn. It supports loading pre-trained
 * models and making efficient predictions.
 *
 * Features:
 * - Thread-safe model loading and prediction
 * - Compatible with scikit-learn GaussianProcessRegressor
 * - Efficient matrix operations using Eigen
 * - Support for RBF kernel with white noise
 */

/**
 * @file gp-predict.h
 * @brief Thread-safe Gaussian Process prediction with kernel support
 * @copyright MIT License
 */

#ifndef GAUSSIAN_PROCESS_H
#define GAUSSIAN_PROCESS_H

#include <Eigen/Dense>
#include "kernels.h"
#include <mutex>
#include <vector>
#include <fstream>
#include <iostream>

/**
 * @brief Thread-safe Gaussian Process prediction class
 *
 * This class loads pre-trained Gaussian Process models from scikit-learn
 * and provides thread-safe prediction capabilities. It uses the RBF kernel
 * with white noise and supports efficient matrix operations via Eigen.
 *
 * Example usage:
 * @code
 * GaussianProcess gp;
 * gp.loadModel("mean.txt", "covariance.txt");
 * 
 * Eigen::VectorXd input(1);
 * input << 2.5;
 * double prediction = gp.predict(input);
 * @endcode
 */
/**
 * @brief Thread-safe Gaussian Process with kernel support
 */
class GaussianProcess {
public:
    GaussianProcess();
    /**
     * @brief Load a pre-trained Gaussian Process model
     * 
     * @param meanFile Path to file containing mean vector
     * @param covFile Path to file containing covariance matrix
     * @return true if model loaded successfully
     * @throws std::runtime_error if files cannot be opened or format is invalid
     * @thread_safety Thread-safe, allows concurrent predictions during load
     */
    bool loadModel(const std::string &meanFile, const std::string &covFile);
    /**
     * @brief Make a prediction for a single input vector
     * 
     * @param x Input vector for prediction
     * @return Predicted value
     * @throws std::runtime_error if model not loaded or input size mismatch
     * @thread_safety Thread-safe, multiple threads can predict concurrently
     */
    double predict(const Eigen::VectorXd &x) const;

    /**
     * @brief Load kernel parameters from file
     */
    bool loadKernel(const std::string& kernelFile);

    /**
     * @brief Compute kernel between training and test points
     */
    Eigen::VectorXd computeKernelVector(const Eigen::VectorXd& x) const;

private:
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
    mutable std::mutex mtx;
    std::shared_ptr<gp::Kernel> kernel_;
    Eigen::MatrixXd X_train_; // Training points
};

#endif // GAUSSIAN_PROCESS_H