/**
 * @file kernels.h
 * @brief Kernel implementations for Gaussian Process regression
 * @copyright MIT License
 */

#ifndef KERNELS_H
#define KERNELS_H

#include <Eigen/Dense>
#include <memory>
#include <cmath>
#include <string>

namespace gp {

// Forward declarations
class Kernel;
class RBFKernel;
class MaternKernel;
class ConstantKernel;
using KernelPtr = std::shared_ptr<Kernel>;

// Operator declarations
KernelPtr operator+(KernelPtr k1, KernelPtr k2);
KernelPtr operator*(KernelPtr k1, KernelPtr k2);

/**
 * @brief Base class for all kernels
 * 
 * Provides the interface for kernel functions used in Gaussian Process regression.
 * Supports both basic kernels (RBF, Matern, Constant) and composite kernels.
 */
class Kernel {
public:
    enum class Type {
        RBF,
        MATERN,
        CONSTANT,
        SUM,
        PRODUCT
    };
    
    virtual ~Kernel() = default;
    virtual Type getType() const = 0;
    virtual std::string toString() const = 0;
    
    /**
     * @brief Compute kernel between two points
     */
    virtual double compute(const Eigen::VectorXd& x1, 
                         const Eigen::VectorXd& x2) const = 0;
    
    /**
     * @brief Compute gram matrix for multiple points
     */
    virtual Eigen::MatrixXd computeGram(const Eigen::MatrixXd& X) const {
        int n = X.rows();
        Eigen::MatrixXd K(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                K(i, j) = compute(X.row(i), X.row(j));
                K(j, i) = K(i, j); // symmetric
            }
        }
        return K;
    }

    /**
     * @brief Create kernel from parameters
     */
    // Forward declaration of the create method - implementation after class definitions
    static KernelPtr create(const std::string& type,
                           const Eigen::VectorXd& params,
                           const std::string& op = "",
                           KernelPtr k1 = nullptr,
                           KernelPtr k2 = nullptr);
};

/**
 * @brief Constant (white noise) kernel
 */
class ConstantKernel : public Kernel {
public:
    explicit ConstantKernel(double constant) : constant_(constant) {}
    
    Type getType() const override { return Type::CONSTANT; }
    
    std::string toString() const override { 
        return "ConstantKernel(constant=" + std::to_string(constant_) + ")"; 
    }
    
    double compute(const Eigen::VectorXd& /* x1 */, 
                  const Eigen::VectorXd& /* x2 */) const override {
        return constant_;
    }
private:
    double constant_;
};

/**
 * @brief RBF (Gaussian) kernel
 */
class RBFKernel : public Kernel {
public:
    explicit RBFKernel(double length_scale) : length_scale_(length_scale) {}
    
    Type getType() const override { return Type::RBF; }
    
    std::string toString() const override { 
        return "RBFKernel(length_scale=" + std::to_string(length_scale_) + ")"; 
    }
    
    double compute(const Eigen::VectorXd& x1, 
                  const Eigen::VectorXd& x2) const override {
        double dist = (x1 - x2).squaredNorm();
        return std::exp(-0.5 * dist / (length_scale_ * length_scale_));
    }
private:
    double length_scale_;
};

/**
 * @brief Matérn kernel
 */
class MaternKernel : public Kernel {
public:
    MaternKernel(double length_scale, double nu) 
        : length_scale_(length_scale), nu_(nu) {}
    
    Type getType() const override { return Type::MATERN; }
    
    std::string toString() const override { 
        return "MaternKernel(length_scale=" + std::to_string(length_scale_) + 
               ", nu=" + std::to_string(nu_) + ")"; 
    }
    
    double compute(const Eigen::VectorXd& x1, 
                  const Eigen::VectorXd& x2) const override {
        double dist = (x1 - x2).norm() / length_scale_;
        
        if (nu_ == 0.5) {
            return std::exp(-dist);
        } else if (nu_ == 1.5) {
            return (1.0 + std::sqrt(3.0) * dist) * 
                   std::exp(-std::sqrt(3.0) * dist);
        } else if (nu_ == 2.5) {
            double sqrt5 = std::sqrt(5.0);
            return (1.0 + sqrt5 * dist + 5.0/3.0 * dist * dist) * 
                   std::exp(-sqrt5 * dist);
        }
        return 0.0; // Other nu values not supported
    }
private:
    double length_scale_;
    double nu_;
};



} // namespace gp

#endif // KERNELS_H
