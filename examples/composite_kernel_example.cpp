/**
 * @file composite_kernel_example.cpp
 * @brief Example using composite kernels in Gaussian Process
 */

#include <iostream>
#include <iomanip>
#include "../gp-predict.h"
#include <Eigen/Dense>

void printPrediction(const GaussianProcess& gp, double x) {
    Eigen::VectorXd input(1);
    input << x;
    double pred = gp.predict(input);
    std::cout << std::fixed << std::setprecision(3)
              << "f(" << x << ") = " << pred << std::endl;
}

int main() {
    try {
        GaussianProcess gp;
        
        // Load model trained with Constant + Matern kernel
        if (!gp.loadModel("data/gp_constant_matern_mean.txt", 
                         "data/gp_constant_matern_covariance.txt")) {
            std::cerr << "Failed to load GP model" << std::endl;
            return -1;
        }
        std::cout << "Model loaded successfully" << std::endl;

        // Make predictions at different points
        std::cout << "\n=== Predictions with Constant + Matern kernel ===" << std::endl;
        for (double x : {0.0, 2.5, 5.0, 7.5, 10.0}) {
            printPrediction(gp, x);
        }

        // Example of extrapolation
        std::cout << "\n=== Extrapolation ===" << std::endl;
        for (double x : {-2.0, 12.0}) {
            printPrediction(gp, x);
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
