#include <gtest/gtest.h>
#include "../polyregression.h"
#include <fstream>
#include <cstdio>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <thread>

class PolynomialRegressionTest : public ::testing::Test {
protected:
    std::string modelFile;
    std::vector<std::pair<Eigen::VectorXd, double>> testData;
    
    void SetUp() override {
        // First try to use the generated test data if available
        std::ifstream genModelFile("polyregression_test_model.txt");
        if (genModelFile.good()) {
            modelFile = "polyregression_test_model.txt";
            genModelFile.close();
            
            // Load test points and expected predictions
            std::ifstream dataFile("polyregression_test_data.txt");
            std::string line;
            while (std::getline(dataFile, line)) {
                std::stringstream ss(line);
                std::string token;
                std::vector<double> values;
                
                while (std::getline(ss, token, ',')) {
                    values.push_back(std::stod(token));
                }
                
                if (values.size() >= 3) { // 2 features + 1 prediction
                    Eigen::VectorXd point(2);
                    point << values[0], values[1];
                    testData.push_back({point, values[2]});
                }
            }
        } else {
            // Create a test model file for a simple polynomial: y = 2 + 3*x1 + 4*x2 + 5*x1^2
            modelFile = "test_poly_model.txt";
            std::ofstream testFile(modelFile);
            testFile << "features: 2\n";
            testFile << "intercept: 2.0\n";
            testFile << "coefficients:\n";
            testFile << "3.0,4.0,5.0\n";  // Coefficients for x1, x2, x1^2
            testFile << "powers:\n";
            testFile << "1,0\n";  // x1
            testFile << "0,1\n";  // x2
            testFile << "2,0\n";  // x1^2
            testFile.close();
            
            // Create test points manually
            Eigen::VectorXd point1(2);
            point1 << 1.0, 2.0;
            testData.push_back({point1, 18.0}); // 2 + 3*1 + 4*2 + 5*1^2 = 18
            
            Eigen::VectorXd point2(2);
            point2 << 2.0, 1.0;
            testData.push_back({point2, 32.0}); // 2 + 3*2 + 4*1 + 5*2^2 = 32
        }
    }

    void TearDown() override {
        // Remove test file if it's our manually created one
        if (modelFile == "test_poly_model.txt") {
            std::remove(modelFile.c_str());
        }
    }
};

TEST_F(PolynomialRegressionTest, LoadModel) {
    PolynomialRegression model;
    ASSERT_NO_THROW(model.loadModel(modelFile));
}

TEST_F(PolynomialRegressionTest, PredictSinglePoint) {
    PolynomialRegression model;
    model.loadModel(modelFile);
    
    // Test predictions for all test points
    for (const auto& testPoint : testData) {
        double prediction = model.predict(testPoint.first);
        EXPECT_NEAR(prediction, testPoint.second, 1e-6) << "Prediction failed for point: " 
            << testPoint.first.transpose();
    }
}

TEST_F(PolynomialRegressionTest, ThreadSafety) {
    PolynomialRegression model;
    model.loadModel(modelFile);
    
    // Test concurrent predictions
    const int numThreads = 10;
    std::vector<std::thread> threads;
    std::vector<double> results(numThreads);
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&model, &results, i, this]() {
            // Use the first test point with small variations
            Eigen::VectorXd x = testData[0].first;
            x(0) += 0.01 * i;
            results[i] = model.predict(x);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify all predictions completed without crashing
    for (int i = 0; i < numThreads; ++i) {
        EXPECT_FALSE(std::isnan(results[i]));
    }
}

TEST_F(PolynomialRegressionTest, PredictBatch) {
    PolynomialRegression model;
    model.loadModel(modelFile);
    
    // Create batch input from test data plus a zero point
    Eigen::MatrixXd X(testData.size() + 1, 2);
    std::vector<double> expectedResults;
    
    for (size_t i = 0; i < testData.size(); ++i) {
        X.row(i) = testData[i].first;
        expectedResults.push_back(testData[i].second);
    }
    
    // Add a zero point (should just predict the intercept)
    X.row(testData.size()) << 0.0, 0.0;
    
    // Get the intercept value from the model file
    double interceptValue = 2.0; // Default fallback
    std::ifstream modelStream(modelFile);
    std::string line;
    while (std::getline(modelStream, line)) {
        if (line.find("intercept:") != std::string::npos) {
            interceptValue = std::stod(line.substr(line.find(":") + 1));
            break;
        }
    }
    expectedResults.push_back(interceptValue);
    
    Eigen::VectorXd predictions = model.predictBatch(X);
    ASSERT_EQ(predictions.size(), static_cast<int>(expectedResults.size()));
    
    for (size_t i = 0; i < expectedResults.size(); ++i) {
        EXPECT_NEAR(predictions(i), expectedResults[i], 1e-6) << "Batch prediction failed at index " << i;
    }
}

TEST_F(PolynomialRegressionTest, InvalidInputSize) {
    PolynomialRegression model;
    model.loadModel(modelFile);
    
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

TEST_F(PolynomialRegressionTest, ConcurrentPredictions) {
    PolynomialRegression model;
    model.loadModel(modelFile);
    
    // Create test inputs
    Eigen::VectorXd x1(2), x2(2);
    x1 << 1.0, 2.0;  // Expected: 18.0
    x2 << 2.0, 1.0;  // Expected: 32.0
    
    // Test concurrent predictions with specific values
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
    
    // Check for expected values
    double expected1 = 0.0, expected2 = 0.0;
    
    // Calculate expected values based on our test data
    for (const auto& testPoint : testData) {
        if ((testPoint.first - x1).norm() < 1e-6) {
            expected1 = testPoint.second;
        }
        if ((testPoint.first - x2).norm() < 1e-6) {
            expected2 = testPoint.second;
        }
    }
    
    EXPECT_NEAR(results[0], expected1, 1e-6);
    EXPECT_NEAR(results[1], expected2, 1e-6);
}
