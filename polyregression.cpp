#include "PolynomialRegression.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

Eigen::VectorXd PolynomialRegression::parseVector(const std::string& line) const {
    std::vector<double> values;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        values.push_back(std::stod(item));
    }
    return Eigen::Map<Eigen::VectorXd>(values.data(), values.size());
}

Eigen::MatrixXi PolynomialRegression::parseMatrix(std::ifstream& file, int rows, int cols) const {
    Eigen::MatrixXi matrix(rows, cols);
    std::string line;
    for (int i = 0; i < rows; ++i) {
        if (!std::getline(file, line)) throw std::runtime_error("Invalid matrix format.");
        std::vector<int> values;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            values.push_back(std::stoi(item));
        }
        if (values.size() != cols) throw std::runtime_error("Column size mismatch.");
        for (int j = 0; j < cols; ++j) matrix(i, j) = values[j];
    }
    return matrix;
}

void PolynomialRegression::loadModel(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Unable to open file.");

    auto newModel = std::make_shared<ModelData>();
    std::string line;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string key;
        ss >> key;

        if (key == "features:") ss >> newModel->features;
        else if (key == "intercept:") ss >> newModel->intercept;
        else if (key == "coefficients:") {
            std::getline(file, line);
            newModel->coefficients = parseVector(line);
        } else if (key == "powers:") {
            newModel->powers = parseMatrix(file, newModel->coefficients.size(), newModel->features);
        }
    }

    if (newModel->features == 0 || newModel->coefficients.size() == 0 || newModel->powers.rows() == 0) {
        throw std::runtime_error("Invalid model file.");
    }

    // Thread-safe model update
    std::unique_lock<std::shared_mutex> lock(mutex);
    model = std::move(newModel);
}

double PolynomialRegression::predict(const Eigen::VectorXd& x) const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto currentModel = model; // Get a reference to the current model
    lock.unlock(); // Release the lock since we have a shared_ptr

    if (x.size() != currentModel->features) {
        throw std::runtime_error("Input size mismatch.");
    }
    
    double result = currentModel->intercept;
    for (int i = 0; i < currentModel->coefficients.size(); ++i) {
        double term = currentModel->coefficients[i];
        for (int j = 0; j < currentModel->features; ++j) {
            if (currentModel->powers(i, j) != 0) {
                term *= std::pow(x[j], currentModel->powers(i, j));
            }
        }
        result += term;
    }
    return result;
}

Eigen::VectorXd PolynomialRegression::predictBatch(const Eigen::MatrixXd& X) const {
    // Get a thread-safe reference to the current model
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto currentModel = model;
    lock.unlock();

    if (X.cols() != currentModel->features) {
        throw std::runtime_error("Input column size mismatch.");
    }

    Eigen::VectorXd predictions(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        predictions[i] = predict(X.row(i));
    }
    return predictions;
}