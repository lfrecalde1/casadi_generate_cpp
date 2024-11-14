#include <iostream>
#include <eigen3/Eigen/Dense>
#include <chrono>
#include "my_package/my_function.h"

// Wrapper function to call the cost function with simplified arguments
int call_cost(float& cost_result, const Eigen::MatrixXf& Q, const Eigen::MatrixXf& x) {
    const casadi_real* arg[2];
    casadi_real* res[1];
    casadi_real w[cost_SZ_W] = {0};
    casadi_int iw[cost_SZ_IW] = {};

    arg[0] = Q.data();
    arg[1] = x.data();
    
    casadi_real output_result = 0.0;
    res[0] = &output_result;

    int status = cost(arg, res, iw, w, 0);

    cost_result = output_result;
    return status;
}

float compute_quadratic_cost(const Eigen::MatrixXf& Q, const Eigen::MatrixXf& x) {
    // Compute the result as a scalar by evaluating the expression
    return (x.transpose() * Q * x)(0, 0);
}

int main() {
    srand(static_cast<unsigned int>(time(0)));
    Eigen::MatrixXf x = Eigen::MatrixXf::Random(1000, 1);
    Eigen::MatrixXf Q = Eigen::MatrixXf::Random(1000, 1000);
    Q = Q.transpose() * Q;  // Ensure Q is symmetric positive definite

    auto start_1 = std::chrono::high_resolution_clock::now();
    float cost_result_1 = compute_quadratic_cost(Q, x);
    auto end_1 = std::chrono::high_resolution_clock::now();

    float cost_result;
    int status;
    for (int i = 0; i < 20; ++i) {
        status = call_cost(cost_result, Q, x);
    }
    auto start = std::chrono::high_resolution_clock::now();
    status = call_cost(cost_result, Q, x);
    auto end = std::chrono::high_resolution_clock::now();

    if (status != 0) {
        std::cerr << "Error calling cost function" << std::endl;
        return -1;
    }

    std::chrono::duration<double> duration = end - start;
    std::chrono::duration<double> duration_1 = end_1 - start_1;

    std::cout << "Quadratic cost (Casadi): " << cost_result << std::endl;
    std::cout << "Execution time (Casadi): " << duration.count() << " seconds" << std::endl;

    std::cout << "Quadratic cost (Eigen): " << cost_result_1 << std::endl;
    std::cout << "Execution time (Eigen): " << duration_1.count() << " seconds" << std::endl;

    if (std::abs(cost_result - cost_result_1) > 1e-3) {
        std::cerr << "Warning: CasADi and Eigen results differ by more than the tolerance!" << std::endl;
    }

    return 0;
}