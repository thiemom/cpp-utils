#include <iostream>
#include <Eigen/Dense>
#include "../polyregression.h"

int main() {
    try {
        PolynomialRegression model;
        model.loadModel("model.txt");  // Load trained sklearn model

        // Example inputs (matching those in Python)
        Eigen::MatrixXd X(3, 2);
        X << 2.0, 3.0,
             1.0, 4.0,
             3.0, 1.0;

        // Single prediction example
        std::cout << "Single prediction for [2.0, 3.0]:\n" 
                  << model.predict(X.row(0)) << "\n\n";

        // Batch predictions
        std::cout << "Batch predictions for all inputs:\n" 
                  << model.predictBatch(X) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
