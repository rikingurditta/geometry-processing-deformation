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
  data.n = V.rows();
  // (negative) cotangent laplacian
  Eigen::SparseMatrix<double> L;
  igl::cotmatrix(V, F, L);
  L *= -1;
  // empty constraints matrix, irrelevant to us
  Eigen::SparseMatrix<double> Aeq(b.rows(), V.rows());
  // precompute system
  igl::min_quad_with_fixed_precompute(L, b, Aeq, true, data);

  // cotangents for each angle in each triangle
  Eigen::MatrixXd C;
  igl::cotmatrix_entries(V, F, C);

  // build K matrix
  K.resize(V.rows(), V.rows() * 3);
  std::vector<Eigen::Triplet<double>> tl;
  for (int f = 0; f < F.rows(); f++) {  // iterate over faces
    for (int i = 0; i < 3; i++) {  // iterate over edges
      int vi = F(f, i);
      int vj = F(f, (i + 1) % 3);
      int opposite = (i + 2) % 3;  // vertex opposite from ij edge
      // e_ij = cot(opposite) * (V(i) - V(j)) / 3
      Eigen::Vector3d e_ij = C(f, opposite) * (V.row(vi) - V.row(vj)) / 3;
      for (int k = 0; k < 3; k++) {  // iterate over vertices
        int vk = F(f, k);
        for (int beta = 0; beta < 3; beta++) {
          tl.emplace_back(vi, vk * 3 + beta, e_ij(beta));
          tl.emplace_back(vj, vk * 3 + beta, -e_ij(beta));
        }
      }
    }
  }
  K.setFromTriplets(tl.begin(), tl.end());
}
