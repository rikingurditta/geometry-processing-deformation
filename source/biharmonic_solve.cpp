#include "biharmonic_solve.h"
#include <igl/min_quad_with_fixed.h>

void biharmonic_solve(
  const igl::min_quad_with_fixed_data<double> & data,
  const Eigen::MatrixXd & bc,
  Eigen::MatrixXd & D)
{
  D = Eigen::MatrixXd::Zero(data.n,3);
  // B is 0 in our case
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(data.n, 3);
  // empty constraints values, irrelevant to us
  Eigen::MatrixXd Beq = Eigen::MatrixXd::Zero(bc.rows(), bc.cols());
  igl::min_quad_with_fixed_solve(data, B, bc, Beq, D);
}
