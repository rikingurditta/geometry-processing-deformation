#include "../include/arap_single_iteration.h"
#include <igl/polar_svd3x3.h>
#include <igl/min_quad_with_fixed.h>

void closest_rotation(const Eigen::Matrix3d &M, Eigen::Matrix3d &R);

void arap_single_iteration(
        const igl::min_quad_with_fixed_data<double> &data,
        const Eigen::SparseMatrix<double> &K,
        const Eigen::MatrixXd &bc,
        Eigen::MatrixXd &U) {
  // i'm really not sure why this is so slow D:
  Eigen::MatrixXd C = K.transpose() * U;
  // find rotations closest to blocks of C
  Eigen::MatrixXd R(C.rows(), 3);
  for (int k = 0; k < C.rows(); k += 3) {
    Eigen::Matrix3d Rk;
    closest_rotation(C.block<3, 3>(k, 0), Rk);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        R(k + i, j) = Rk(i, j);  // copy Rk to R
      }
    }
  }
  // empty constraints values, irrelevant to us
  Eigen::MatrixXd Beq(bc.rows(), bc.cols());
  // have to solve with B = -KR, probably because we negated L earlier
  igl::min_quad_with_fixed_solve(data, -K * R, bc, Beq, U);
}

// find closest rotation to given matrix using singular value decomposition
void closest_rotation(const Eigen::Matrix3d &M, Eigen::Matrix3d &R) {
  // find SVD of M to get rotations
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullV | Eigen::ComputeFullU);
  const Eigen::Matrix3d &U = svd.matrixU();
  const Eigen::Matrix3d &V = svd.matrixV();
  // optimal omega has 1, 1, det(UV^T) on diagonal
  Eigen::Matrix3d omega_star = Eigen::Matrix3d::Identity();
  // det(UV^T) = det(U)det(V^T) = det(U)det(V)
  omega_star(2, 2) = U.determinant() * V.determinant();
  // compute R based on optimization formula
  // return transpose to match use in later computations
  R = (U * omega_star * V.transpose()).transpose();
}
