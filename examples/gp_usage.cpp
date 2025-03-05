/**
 * @file gp_usage.cpp
 * @brief Example usage of the GaussianProcess class
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

#include <iostream>
#include <vector>
#include <thread>
#include "gp-predict.h"
#include <Eigen/Dense>

/**
 * @brief Example of single-point prediction
 * 
 * @param gp Reference to GaussianProcess instance
 */
void singlePrediction(const GaussianProcess& gp) {
    try {
        Eigen::VectorXd testPoint(1);
        testPoint << 5.0;  // Predict at x = 5.0

        double prediction = gp.predict(testPoint);
        std::cout << "Single prediction at x = 5.0: " << prediction << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in single prediction: " << e.what() << std::endl;
    }
}

/**
 * @brief Example of concurrent predictions from multiple threads
 * 
 * @param gp Reference to GaussianProcess instance
 */
void concurrentPredictions(const GaussianProcess& gp) {
    try {
        // Create multiple threads making predictions
        std::vector<std::thread> threads;
        std::vector<double> points = {2.0, 4.0, 6.0, 8.0};
        
        for (double x : points) {
            threads.emplace_back([&gp, x]() {
                Eigen::VectorXd testPoint(1);
                testPoint << x;
                double prediction = gp.predict(testPoint);
                std::cout << "Thread prediction at x = " << x 
                         << ": " << prediction << std::endl;
            });
        }
        
        // Wait for all predictions to complete
        for (auto& thread : threads) {
            thread.join();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in concurrent predictions: " << e.what() << std::endl;
    }
}

int main() {
    try {
        // Create and load the Gaussian Process model
        GaussianProcess gp;
        
        if (!gp.loadModel("data/mean.txt", "data/covariance.txt")) {
            std::cerr << "Failed to load GP model" << std::endl;
            return -1;
        }
        std::cout << "Model loaded successfully" << std::endl;

        // Demonstrate single prediction
        std::cout << "\n=== Single Prediction Example ===" << std::endl;
        singlePrediction(gp);

        // Demonstrate concurrent predictions
        std::cout << "\n=== Concurrent Predictions Example ===" << std::endl;
        concurrentPredictions(gp);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
