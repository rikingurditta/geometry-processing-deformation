#include "biharmonic_precompute.h"
#include <igl/min_quad_with_fixed.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>

void biharmonic_precompute(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  const Eigen::VectorXi & b,
  igl::min_quad_with_fixed_data<double> & data)
{
  // mass matrix
  Eigen::SparseMatrix<double> M;
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
  // inverse mass matrix
  Eigen::SparseMatrix<double> Minv;
  igl::invert_diag(M, Minv);
  // cotangent laplacian
  Eigen::SparseMatrix<double> L;
  igl::cotmatrix(V, F, L);
  // bi-laplacian, more or less
  Eigen::SparseMatrix<double> Q = L.transpose() * Minv * L;
  // empty constraints matrix, irrelevant to us
  Eigen::SparseMatrix<double> Aeq(b.rows(), V.rows());
  // precompute system
  igl::min_quad_with_fixed_precompute(Q, b, Aeq, true, data);
  data.n = V.rows();
}
