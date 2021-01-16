#include <ceres/ceres.h>
#include "camera_info.hpp"
#include <Eigen/Dense>

using namespace Eigen;

class ReprojectionFactor : public ceres::SizedCostFunction<2, 7, 3> {
  public:
    ReprojectionFactor(double ox, double oy, CameraInfo info);

    virtual bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const;

  private:
    Matrix<double, 2, 3> K; // camera matrix
    Vector2d obs; // observation, homogeneous coordinates
    CameraInfo camera_info;
};
