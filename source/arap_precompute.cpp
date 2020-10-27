#include "../include/arap_precompute.h"
#include <igl/min_quad_with_fixed.h>
#include <igl/cotmatrix.h>
#include <igl/cotmatrix_entries.h>

void arap_precompute(
        const Eigen::MatrixXd &V,
        const Eigen::MatrixXi &F,
        const Eigen::VectorXi &b,
        igl::min_quad_with_fixed_data<double> &data,
        Eigen::SparseMatrix<double> &K) {
  // (negative) cotangent laplacian
  Eigen::SparseMatrix<double> L;
  igl::cotmatrix(V, F, L);
  L *= -1;
  // empty constraints matrix, irrelevant to us
  Eigen::SparseMatrix<double> Aeq(b.rows(), V.rows());
  igl::min_quad_with_fixed_precompute(L, b, Aeq, true, data);
  // precompute system
  // cotangents for each angle in triangle
  Eigen::MatrixXd C;
  igl::cotmatrix_entries(V, F, C);

  // calculate K using formula in handout
  K.resize(V.rows(), V.rows() * 3);
  std::vector<Eigen::Triplet<double>> tripletList;
  for (int f = 0; f < F.rows(); f++) {
    for (int i = 0; i < 3; i++) {
      int vi = F(f, i);
      int vj = F(f, (i + 1) % 3);
      int vo = F(f, (i + 2) % 3);  // vertex opposite to ij edge
      for (int k = 0; k < 3; k++) {
        int vk = F(f, k);
        Eigen::Vector3d e_ij = C(f, vo) * V.row(vi) - V.row(vj);
        for (int beta = 0; beta < 3; beta++) {
          tripletList.emplace_back(vi, 3 * vk + beta, e_ij(beta));
          tripletList.emplace_back(vk, 3 * vk + beta, -e_ij(beta));
        }
      }
    }
  }
  K.setFromTriplets(tripletList.begin(), tripletList.end());
}
