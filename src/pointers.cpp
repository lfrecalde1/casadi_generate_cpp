#include <iostream>
#include <eigen3/Eigen/Dense>
#include <chrono>
#include "my_package/my_function_cost.h"
#include "my_package/my_function_delta.h"
#include <matplot/matplot.h>
#include <cmath>
#include <vector>

namespace plt = matplot;

// Wrapper function to call the cost function with simplified arguments
int call_cost(float& cost_result, const Eigen::MatrixXf& Q, const Eigen::MatrixXf& c, const Eigen::MatrixXf& x, const Eigen::MatrixXf& xd) {
    const casadi_real* arg[4];
    casadi_real* res[1];
    casadi_real w[cost_SZ_W] = {0};
    casadi_int iw[cost_SZ_IW] = {};

    arg[0] = Q.data();
    arg[1] = c.data();
    arg[2] = x.data();
    arg[3] = xd.data();
    
    casadi_real output_result = 0.0;
    res[0] = &output_result;

    int status = cost(arg, res, iw, w, 0);

    cost_result = output_result;
    return status;
}

int call_delta(Eigen::MatrixXf& delta_result, const Eigen::MatrixXf& Q, const Eigen::MatrixXf& c, const Eigen::MatrixXf& x, const Eigen::MatrixXf& xd) {
    // Ensure delta_result is the correct size (2x1)
    delta_result.resize(2, 1);

    const casadi_real* arg[4];
    casadi_real* res[1];
    casadi_real w[delta_x_SZ_W] = {0};
    casadi_int iw[delta_x_SZ_IW] = {};

    // Prepare input arguments
    arg[0] = Q.data();
    arg[1] = c.data();
    arg[2] = x.data();
    arg[3] = xd.data();
    
    // Prepare output result to store a 2x1 vector
    casadi_real output_result[2] = {0.0, 0.0};
    res[0] = output_result;

    // Call the CasADi function
    int status = delta_x(arg, res, iw, w, 0);

    // Assign the result to delta_result (an Eigen::MatrixXf of size 2x1)
    delta_result(0, 0) = output_result[0];
    delta_result(1, 0) = output_result[1];

    return status;
}

float compute_quadratic_cost(const Eigen::MatrixXf& Q, const Eigen::MatrixXf& x) {
    // Compute the result as a scalar by evaluating the expression
    return (x.transpose() * Q * x)(0, 0);
}

int main() {
    srand(static_cast<unsigned int>(time(0)));
    // Eigen variables
    Eigen::MatrixXf x = Eigen::MatrixXf::Random(2, 5000);
    x = x * 20.0f; // Scale to [-10, 10]
    Eigen::MatrixXf Q = Eigen::MatrixXf::Zero(2, 2);
    Eigen::MatrixXf result = Eigen::MatrixXf::Zero(1, 5000);
    Eigen::MatrixXf c = Eigen::MatrixXf::Zero(2, 1);
    Eigen::MatrixXf xd = Eigen::MatrixXf::Zero(2, 1);
    Eigen::MatrixXf delta = Eigen::MatrixXf::Zero(2, 1);
    Eigen::MatrixXf guess = Eigen::MatrixXf::Zero(2, 50);
    Eigen::MatrixXf cost_guess = Eigen::MatrixXf::Zero(1, 50);
    Q = Q.transpose() * Q;

    // Modify values using pointers
    float* c_ptr = c.data();
    float* Q_ptr = Q.data();
    float* xd_ptr = xd.data();

    // Set values of c using pointer
    *(c_ptr) = 0.0;      // Equivalent to c(0, 0) = 0.0
    *(c_ptr + 1) = 0.0;  // Equivalent to c(1, 0) = 0.0

    // Set values of xd using pointer
    *(xd_ptr) = -2.0;      // Equivalent to xd(0, 0) = 5.0
    *(xd_ptr + 1) = -2.0; // Equivalent to xd(1, 0) = -5.0

    *(Q_ptr + 0) = 1.0f; // Equivalent to Q(0, 0) = 1.0
    *(Q_ptr + 3) = 1.0f; // Equivalent to Q(1, 1) = 1.0

    // initialize the guess value
    guess(0, 0) = -15;
    guess(1, 0) = 15;

    // Pointer results
    float* result_ptr = result.data();
    float* cost_ptr = cost_guess.data();
    float* delta_ptr = delta.data();

    // Compute cost using Eigen
    auto start_1 = std::chrono::high_resolution_clock::now();
    float cost_result_1 = compute_quadratic_cost(Q, x);
    auto end_1 = std::chrono::high_resolution_clock::now();

    // Alternative variables
    float cost_result;
    int status;
    int status_1;
    int status_2;

    // Checking the behavior of the casadi function 
    for (int i = 0; i < x.cols(); ++i) {
        status = call_cost(*(result_ptr + i), Q, c, x.col(i), xd);
        //std::chrono::duration<double> duration = end - start;
        //std::cout << "Result[" << i << "] = " << *(result_ptr + i) << std::endl;
        //std::cout << "Execution time (Casadi): " << duration.count() << " seconds" << std::endl;
        //std::cout << "Number (Casadi): " << i << std::endl;
    }

    // Compute optimal variable
    for (int k = 0; k < guess.cols()-1; ++k) {
        status_2 = call_cost(*(cost_ptr + k), Q, c, guess.col(k), xd);
        auto start = std::chrono::high_resolution_clock::now();
        status_1 = call_delta(delta, Q, c, guess.col(k), xd);
        guess.col(k + 1) = guess.col(k) + 0.2*delta;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Execution time Optimization: " << duration.count() << " seconds" << std::endl;
        //std::cout << "Result guess[" << k << "] = " << *(cost_ptr + k) << std::endl;
        //std::cout << "Guess[" << k << "] = " << guess.col(k) << std::endl;
        //std::cout << "Delta Value[" << k << "] = " << delta << std::endl;
    }


    // Eigen Section to use plot
    Eigen::VectorXf result_vector = result.reshaped<Eigen::RowMajor>().cast<float>();
    Eigen::VectorXf x1 = x.row(0);
    Eigen::VectorXf x2 = x.row(1);

    // New Values for the guess variables
    // Extract cost_guess from the beginning to end - 1
    Eigen::VectorXf cost_guess_vector = cost_guess.block(0, 0, 1, cost_guess.cols() - 1).transpose();

    // Extract x1_guess from the beginning to end - 1
    Eigen::VectorXf x1_guess = guess.block(0, 0, 1, guess.cols() - 1).transpose();

    // Extract x2_guess from the beginning to end - 1
    Eigen::VectorXf x2_guess = guess.block(1, 0, 1, guess.cols() - 1).transpose();

    plt::figure();
    plt::plot3(x1, x2, result_vector, "c*", x1_guess, x2_guess, cost_guess_vector, "rx");

    plt::xlabel("x1");
    plt::ylabel("x2");
    plt::zlabel("f(x)");

    plt::show();

    return 0;
}