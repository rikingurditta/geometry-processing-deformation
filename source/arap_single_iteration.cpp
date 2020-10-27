#include "../include/arap_single_iteration.h"
#include <igl/polar_svd3x3.h>
#include <igl/min_quad_with_fixed.h>

void arap_single_iteration(
  const igl::min_quad_with_fixed_data<double> & data,
  const Eigen::SparseMatrix<double> & K,
  const Eigen::MatrixXd & bc,
  Eigen::MatrixXd & U)
{
  Eigen::MatrixXd C = K.transpose() * U;
  Eigen::MatrixXd R(C.rows(), C.cols());
  for (int k = 0; k < U.rows(); k++) {
    Eigen::Matrix3d Ck = C.block<3, 3>(k * 3, 0);
    Eigen::Matrix3d Rk;
    igl::polar_svd3x3(Ck, Rk);
    R.block<3, 3>(k * 3, 0) = Rk;
  }
  Eigen::MatrixXd Beq = Eigen::MatrixXd::Zero(bc.rows(), bc.cols());
  igl::min_quad_with_fixed_solve(data, K * R, bc, Beq, U);
}
