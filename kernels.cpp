/**
 * @file kernels.cpp
 * @brief Implementation of kernel operators for Gaussian Process regression
 * @copyright MIT License
 */

#include "kernels.h"

namespace gp {

// Implementation of composite kernels

/**
 * @brief Sum kernel class - adds the results of two kernels
 */
class SumKernel : public Kernel {
public:
    SumKernel(KernelPtr k1, KernelPtr k2) : k1_(k1), k2_(k2) {}
    
    Type getType() const override { return Type::SUM; }
    
    std::string toString() const override { 
        return "SumKernel(" + k1_->toString() + " + " + k2_->toString() + ")"; 
    }
    
    double compute(const Eigen::VectorXd& x1, 
                  const Eigen::VectorXd& x2) const override {
        return k1_->compute(x1, x2) + k2_->compute(x1, x2);
    }
    
private:
    KernelPtr k1_;
    KernelPtr k2_;
};

/**
 * @brief Product kernel class - multiplies the results of two kernels
 */
class ProductKernel : public Kernel {
public:
    ProductKernel(KernelPtr k1, KernelPtr k2) : k1_(k1), k2_(k2) {}
    
    Type getType() const override { return Type::PRODUCT; }
    
    std::string toString() const override { 
        return "ProductKernel(" + k1_->toString() + " * " + k2_->toString() + ")"; 
    }
    
    double compute(const Eigen::VectorXd& x1, 
                  const Eigen::VectorXd& x2) const override {
        return k1_->compute(x1, x2) * k2_->compute(x1, x2);
    }
    
private:
    KernelPtr k1_;
    KernelPtr k2_;
};

// Implementation of kernel operators
KernelPtr operator+(KernelPtr k1, KernelPtr k2) {
    return std::make_shared<SumKernel>(k1, k2);
}

KernelPtr operator*(KernelPtr k1, KernelPtr k2) {
    return std::make_shared<ProductKernel>(k1, k2);
}

// Implementation of the static create method
KernelPtr Kernel::create(const std::string& type,
                       const Eigen::VectorXd& params,
                       const std::string& op,
                       KernelPtr k1,
                       KernelPtr k2) {
    if (type == "rbf") {
        if (params.size() != 1) {
            throw std::runtime_error("RBF kernel requires 1 parameter (length_scale)");
        }
        return std::make_shared<RBFKernel>(params(0));
    } else if (type == "matern") {
        if (params.size() != 2) {
            throw std::runtime_error("Matern kernel requires 2 parameters (length_scale, nu)");
        }
        return std::make_shared<MaternKernel>(params(0), params(1));
    } else if (type == "constant") {
        if (params.size() != 1) {
            throw std::runtime_error("Constant kernel requires 1 parameter (constant)");
        }
        return std::make_shared<ConstantKernel>(params(0));
    } else if (type == "composite") {
        if (!k1 || !k2) {
            throw std::runtime_error("Composite kernel requires two sub-kernels");
        }
        if (op == "sum") {
            return k1 + k2;
        } else if (op == "product") {
            return k1 * k2;
        }
        throw std::runtime_error("Unsupported composite operation: " + op);
    }
    throw std::runtime_error("Unsupported kernel type: " + type);
}

} // namespace gp
