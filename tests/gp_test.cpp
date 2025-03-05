#include <gtest/gtest.h>
#include "../gp-predict.h"
#include <fstream>
#include <cstdio>
#include <Eigen/Dense>
#include <cmath>
#include <thread>

class GaussianProcessTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test model files for a simple GP model
        // Mean file - contains the mean values at training points
        std::ofstream meanFile("test_gp_mean.txt");
        meanFile << "0.5\n";  // Mean value at training point
        meanFile.close();
        
        // Covariance file - contains the covariance matrix
        std::ofstream covFile("test_gp_cov.txt");
        covFile << "1.0\n";  // Diagonal element of covariance matrix
        covFile.close();
        
        // Kernel file - format should be just 'rbf' not 'kernel: rbf'
        std::ofstream kernelFile("test_gp_mean.txt.kernel");
        kernelFile << "rbf\n";  // Kernel type
        kernelFile << "1.0\n";  // Length scale parameter
        kernelFile.close();
        
        // Training data file - required by loadModel
        std::ofstream trainFile("test_gp_mean.txt.train");
        trainFile << "1.0 2.0\n";  // Single training point
        trainFile.close();
        
        // Print debug info
        std::cout << "Created test files for GP model" << std::endl;
    }

    void TearDown() override {
        // Remove test files
        std::remove("test_gp_mean.txt");
        std::remove("test_gp_cov.txt");
        std::remove("test_gp_mean.txt.kernel");
        std::remove("test_gp_mean.txt.train");
    }
};

TEST_F(GaussianProcessTest, LoadModel) {
    GaussianProcess gp;
    ASSERT_TRUE(gp.loadModel("test_gp_mean.txt", "test_gp_cov.txt"));
    ASSERT_TRUE(gp.loadKernel("test_gp_mean.txt.kernel"));
}

TEST_F(GaussianProcessTest, PredictSinglePoint) {
    GaussianProcess gp;
    // Load the model and print success/failure
    bool modelLoaded = gp.loadModel("test_gp_mean.txt", "test_gp_cov.txt");
    std::cout << "Model loaded: " << (modelLoaded ? "successfully" : "failed") << std::endl;
    
    // Load the kernel and print success/failure
    bool kernelLoaded = gp.loadKernel("test_gp_mean.txt.kernel");
    std::cout << "Kernel loaded: " << (kernelLoaded ? "successfully" : "failed") << std::endl;
    
    // Test prediction for a point identical to the training point
    Eigen::VectorXd x(2);
    x << 1.0, 2.0;
    double prediction = gp.predict(x);
    std::cout << "Prediction for training point: " << prediction << std::endl;
    
    // Since we're predicting at the exact training point and using RBF kernel,
    // the prediction should be exactly the mean value
    EXPECT_NEAR(prediction, 0.5, 0.1);
}

TEST_F(GaussianProcessTest, ThreadSafety) {
    GaussianProcess gp;
    // Load the model and print success/failure
    bool modelLoaded = gp.loadModel("test_gp_mean.txt", "test_gp_cov.txt");
    std::cout << "Model loaded: " << (modelLoaded ? "successfully" : "failed") << std::endl;
    
    // Load the kernel and print success/failure
    bool kernelLoaded = gp.loadKernel("test_gp_mean.txt.kernel");
    std::cout << "Kernel loaded: " << (kernelLoaded ? "successfully" : "failed") << std::endl;
    
    // Test concurrent predictions
    const int numThreads = 10;
    std::vector<std::thread> threads;
    std::vector<double> results(numThreads);
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&gp, &results, i]() {
            Eigen::VectorXd x(2);
            x << 1.0, 2.0;  // Use the exact training point for all threads
            results[i] = gp.predict(x);
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
