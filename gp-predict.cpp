/**
 * @file gp-predict.cpp
 * @brief Implementation of the GaussianProcess class
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
 */

#include "gp-predict.h"

/**
 * @brief Default constructor
 */
GaussianProcess::GaussianProcess() {}

/**
 * @brief Load a pre-trained Gaussian Process model from files
 *
 * Loads the mean vector and covariance matrix from separate files.
 * The files should be in the format exported by sklearn_export.py.
 * Thread safety is ensured through mutex locking.
 *
 * @param meanFile Path to the mean vector file
 * @param covFile Path to the covariance matrix file
 * @return true if loading successful
 */
bool GaussianProcess::loadModel(const std::string &meanFile, const std::string &covFile) {
    std::lock_guard<std::mutex> lock(mtx);

    // Load training points and kernel parameters
    std::ifstream trainStream(meanFile + ".train");
    if (!trainStream.is_open()) {
        std::cerr << "Error loading training data!" << std::endl;
        return false;
    }

    std::vector<std::vector<double>> trainData;
    std::string line;
    while (std::getline(trainStream, line)) {
        std::istringstream iss(line);
        std::vector<double> point;
        double val;
        while (iss >> val) {
            point.push_back(val);
        }
        trainData.push_back(point);
    }
    
    int n_points = trainData.size();
    int n_features = trainData[0].size();
    X_train_.resize(n_points, n_features);
    for (int i = 0; i < n_points; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X_train_(i, j) = trainData[i][j];
        }
    }

    // Load kernel parameters
    if (!loadKernel(meanFile + ".kernel")) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx);

    std::ifstream meanStream(meanFile);
    std::ifstream covStream(covFile);

    if (!meanStream.is_open() || !covStream.is_open()) {
        std::cerr << "Error loading model files!" << std::endl;
        return false;
    }

    std::vector<double> meanData;
    double val;

    while (meanStream >> val) {
        meanData.push_back(val);
    }
    mean = Eigen::VectorXd::Map(meanData.data(), meanData.size());

    std::vector<double> covData;
    int size = meanData.size();
    while (covStream >> val) {
        covData.push_back(val);
    }
    covariance = Eigen::MatrixXd::Map(covData.data(), size, size);

    meanStream.close();
    covStream.close();

    std::cout << "Model loaded successfully." << std::endl;
    return true;
}

/**
 * @brief Make a prediction using the loaded GP model
 *
 * Uses the RBF kernel to compute the prediction for new input.
 * Thread safety is ensured through shared mutex locking.
 *
 * @param x Input vector for prediction
 * @return Predicted value
 * @throws std::runtime_error if model not loaded
 */
bool GaussianProcess::loadKernel(const std::string& kernelFile) {
    std::ifstream kernelStream(kernelFile);
    if (!kernelStream.is_open()) {
        std::cerr << "Error loading kernel parameters!" << std::endl;
        return false;
    }

    std::string kernelType;
    std::getline(kernelStream, kernelType);

    std::vector<double> params;
    double param;
    while (kernelStream >> param) {
        params.push_back(param);
    }

    Eigen::Map<Eigen::VectorXd> paramVec(params.data(), params.size());
    kernel_ = gp::Kernel::create(kernelType, paramVec);
    
    return kernel_ != nullptr;
}

Eigen::VectorXd GaussianProcess::computeKernelVector(const Eigen::VectorXd& x) const {
    int n = X_train_.rows();
    Eigen::VectorXd k(n);
    for (int i = 0; i < n; ++i) {
        k(i) = kernel_->compute(x, X_train_.row(i));
    }
    return k;
}

double GaussianProcess::predict(const Eigen::VectorXd &x) const {
    std::lock_guard<std::mutex> lock(mtx);

    if (mean.size() == 0 || covariance.size() == 0) {
        std::cerr << "Model is not loaded!" << std::endl;
        return 0.0;
    }

    Eigen::VectorXd k = computeKernelVector(x);
    Eigen::VectorXd alpha = covariance.llt().solve(mean);

    return k.dot(alpha);
}