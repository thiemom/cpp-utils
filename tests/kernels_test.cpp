#include <gtest/gtest.h>
#include "../kernels.h"
#include <Eigen/Dense>
#include <memory>

namespace gp {

TEST(KernelsTest, RBFKernel) {
    // Create RBF kernel with length_scale = 1.0
    auto rbf = std::make_shared<RBFKernel>(1.0);
    
    // Test kernel computation for identical points
    Eigen::VectorXd x1(2);
    x1 << 1.0, 2.0;
    
    // K(x,x) should be 1.0 for RBF kernel
    EXPECT_DOUBLE_EQ(rbf->compute(x1, x1), 1.0);
    
    // Test kernel computation for different points
    Eigen::VectorXd x2(2);
    x2 << 2.0, 3.0;
    
    // Distance between points is sqrt(2)
    // K(x1,x2) = exp(-0.5 * (sqrt(2))^2 / 1.0^2) = exp(-1.0) ≈ 0.368
    double expected = std::exp(-1.0);
    EXPECT_NEAR(rbf->compute(x1, x2), expected, 1e-6);
}

TEST(KernelsTest, MaternKernel) {
    // Create Matern kernel with length_scale = 1.0 and nu = 1.5
    auto matern = std::make_shared<MaternKernel>(1.0, 1.5);
    
    // Test kernel computation for identical points
    Eigen::VectorXd x1(2);
    x1 << 1.0, 2.0;
    
    // K(x,x) should be 1.0 for Matern kernel
    EXPECT_DOUBLE_EQ(matern->compute(x1, x1), 1.0);
    
    // Test kernel computation for different points
    Eigen::VectorXd x2(2);
    x2 << 2.0, 3.0;
    
    // For nu=1.5, the Matern kernel has a specific form
    // We're just checking it returns a reasonable value between 0 and 1
    double result = matern->compute(x1, x2);
    EXPECT_GT(result, 0.0);
    EXPECT_LT(result, 1.0);
}

TEST(KernelsTest, ConstantKernel) {
    // Create Constant kernel with value = 2.5
    auto constant = std::make_shared<ConstantKernel>(2.5);
    
    // Test kernel computation for any points
    Eigen::VectorXd x1(2);
    x1 << 1.0, 2.0;
    
    Eigen::VectorXd x2(2);
    x2 << 3.0, 4.0;
    
    // K(x1,x2) should always be the constant value
    EXPECT_DOUBLE_EQ(constant->compute(x1, x2), 2.5);
    EXPECT_DOUBLE_EQ(constant->compute(x1, x1), 2.5);
}

TEST(KernelsTest, KernelOperators) {
    auto rbf = std::make_shared<RBFKernel>(1.0);
    auto constant = std::make_shared<ConstantKernel>(2.0);
    
    // Test sum kernel
    auto sum = rbf + constant;
    
    Eigen::VectorXd x1(2);
    x1 << 1.0, 2.0;
    
    Eigen::VectorXd x2(2);
    x2 << 1.0, 2.0;
    
    // Sum kernel should add the results
    EXPECT_DOUBLE_EQ(sum->compute(x1, x2), rbf->compute(x1, x2) + constant->compute(x1, x2));
    
    // Test product kernel
    auto product = rbf * constant;
    
    // Product kernel should multiply the results
    EXPECT_DOUBLE_EQ(product->compute(x1, x2), rbf->compute(x1, x2) * constant->compute(x1, x2));
}

TEST(KernelsTest, CreateMethod) {
    // Test creating RBF kernel
    Eigen::VectorXd rbfParams(1);
    rbfParams << 1.5;
    auto rbf = Kernel::create("rbf", rbfParams);
    EXPECT_EQ(rbf->getType(), Kernel::Type::RBF);
    
    // Test creating Matern kernel
    Eigen::VectorXd maternParams(2);
    maternParams << 1.0, 2.5;
    auto matern = Kernel::create("matern", maternParams);
    EXPECT_EQ(matern->getType(), Kernel::Type::MATERN);
    
    // Test creating Constant kernel
    Eigen::VectorXd constantParams(1);
    constantParams << 3.0;
    auto constant = Kernel::create("constant", constantParams);
    EXPECT_EQ(constant->getType(), Kernel::Type::CONSTANT);
    
    // Test creating composite kernel
    auto composite = Kernel::create("composite", Eigen::VectorXd(), "sum", rbf, constant);
    EXPECT_EQ(composite->getType(), Kernel::Type::SUM);
}

} // namespace gp
