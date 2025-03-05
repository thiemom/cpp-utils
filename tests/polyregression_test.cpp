#include <gtest/gtest.h>
#include "../polyregression.h"
#include <fstream>
#include <cstdio>
#include <Eigen/Dense>
#include <cmath>

class PolynomialRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test model file for a simple polynomial: y = 2 + 3*x1 + 4*x2 + 5*x1^2
        std::ofstream testFile("test_poly_model.txt");
        testFile << "features: 2\n";
        testFile << "intercept: 2.0\n";
        testFile << "coefficients:\n";
        testFile << "3.0,4.0,5.0\n";  // Coefficients for x1, x2, x1^2
        testFile << "powers:\n";
        testFile << "1,0\n";  // x1
        testFile << "0,1\n";  // x2
        testFile << "2,0\n";  // x1^2
        testFile.close();
    }

    void TearDown() override {
        // Remove test file
        std::remove("test_poly_model.txt");
    }
};

TEST_F(PolynomialRegressionTest, LoadModel) {
    PolynomialRegression model;
    ASSERT_NO_THROW(model.loadModel("test_poly_model.txt"));
}

TEST_F(PolynomialRegressionTest, PredictSinglePoint) {
    PolynomialRegression model;
    model.loadModel("test_poly_model.txt");
    
    // Test prediction for x1=1, x2=2
    // Expected: y = 2 + 3*1 + 4*2 + 5*1^2 = 2 + 3 + 8 + 5 = 18
    Eigen::VectorXd x(2);
    x << 1.0, 2.0;
    double prediction = model.predict(x);
    EXPECT_DOUBLE_EQ(prediction, 18.0);
    
    // Test prediction for x1=2, x2=1
    // Expected: y = 2 + 3*2 + 4*1 + 5*2^2 = 2 + 6 + 4 + 20 = 32
    x << 2.0, 1.0;
    prediction = model.predict(x);
    EXPECT_DOUBLE_EQ(prediction, 32.0);
}

TEST_F(PolynomialRegressionTest, PredictBatch) {
    PolynomialRegression model;
    model.loadModel("test_poly_model.txt");
    
    // Test batch prediction
    Eigen::MatrixXd X(3, 2);
    X << 1.0, 2.0,   // y = 18
         2.0, 1.0,   // y = 32
         0.0, 0.0;   // y = 2 (just the intercept)
    
    Eigen::VectorXd predictions = model.predictBatch(X);
    ASSERT_EQ(predictions.size(), 3);
    EXPECT_DOUBLE_EQ(predictions(0), 18.0);
    EXPECT_DOUBLE_EQ(predictions(1), 32.0);
    EXPECT_DOUBLE_EQ(predictions(2), 2.0);
}

TEST_F(PolynomialRegressionTest, InvalidInputSize) {
    PolynomialRegression model;
    model.loadModel("test_poly_model.txt");
    
    // Test with wrong input size
    Eigen::VectorXd x(3);  // Model expects 2 features
    x << 1.0, 2.0, 3.0;
    
    EXPECT_THROW(model.predict(x), std::runtime_error);
}

TEST_F(PolynomialRegressionTest, InvalidModelFile) {
    PolynomialRegression model;
    
    // Test with non-existent file
    EXPECT_THROW(model.loadModel("nonexistent_file.txt"), std::runtime_error);
    
    // Create invalid model file (missing powers)
    std::ofstream testFile("invalid_model.txt");
    testFile << "features: 2\n";
    testFile << "intercept: 2.0\n";
    testFile << "coefficients:\n";
    testFile << "3.0,4.0,5.0\n";
    // Missing powers section
    testFile.close();
    
    EXPECT_THROW(model.loadModel("invalid_model.txt"), std::runtime_error);
    std::remove("invalid_model.txt");
}

TEST_F(PolynomialRegressionTest, ThreadSafety) {
    PolynomialRegression model;
    model.loadModel("test_poly_model.txt");
    
    // Create test inputs
    Eigen::VectorXd x1(2), x2(2);
    x1 << 1.0, 2.0;  // Expected: 18.0
    x2 << 2.0, 1.0;  // Expected: 32.0
    
    // Test concurrent predictions
    std::vector<std::thread> threads;
    std::vector<double> results(2, 0.0);
    
    threads.emplace_back([&model, &x1, &results]() {
        results[0] = model.predict(x1);
    });
    
    threads.emplace_back([&model, &x2, &results]() {
        results[1] = model.predict(x2);
    });
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_DOUBLE_EQ(results[0], 18.0);
    EXPECT_DOUBLE_EQ(results[1], 32.0);
}
