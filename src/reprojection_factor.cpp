#include "reprojection_factor.hpp"

ReprojectionFactor::ReprojectionFactor(double ox, double oy, CameraInfo info) {
  obs = Vector2d(ox, oy);
  K << info.focal, 0         , info.cx,
       0         , info.focal, info.cy;
}

bool ReprojectionFactor::Evaluate(double const* const* parameters,
                                  double* residuals,
                                  double** jacobians) const {

  // Associate residuals and parameters with eigen vectors
  Map<const Vector4d> q(parameters[0]); // orientation of world frame wrt camera (wxyz)
  Map<const Vector3d> t(parameters[0] + 4); // translation of world frame wrt camera

  Map<const Vector3d> p(parameters[1]); // 3d feature position in world frame

  Map<Vector2d> resid(residuals);

  // Calculate things we'll need later

  Matrix3d ss;
  ss << 0    , -q(3), q(2),
        q(3) , 0    , -q(1),
        -q(2), q(1) , 0;

  Matrix3d ll = ss + q(0)*Matrix3d::Identity();

  Matrix3d LR = q.tail<3>()*(q.tail<3>().transpose()) + ll*ll;

  Vector3d gamma = LR*p + t;

  double psi = 1/(2*((q(1)*q(3) - q(0)*q(2))*p(0) + (q(2)*q(3) + q(0)*q(1))*p(1)) +
                  (q(0)*q(0) + q(3)*q(3) - q(1)*q(1) - q(2)*q(2))*p(2) + t(2));
  double npsi2 = -psi*psi;

  // Calculate residual
  resid = K*psi*gamma - obs;

  // Compute jacobians
  if (jacobians) {
    // Jacobian wrt pose
    if (jacobians[0]) {
      // Orientation
      Map<Matrix<double, 2, 4, RowMajor>> dr_dq(jacobians[0]);

      Matrix<double, 3, 4> dgamma_dq;
      dgamma_dq.block<3, 1>(0, 0) = 2*(q.tail<3>().cross(p) + q(0)*p);

      double qtp = 2*q.tail<3>().dot(p);

      dgamma_dq(0, 1) = qtp;
      dgamma_dq(1, 1) = 2*(q(2)*p(0) - q(1)*p(1) - q(0)*p(2));
      dgamma_dq(2, 1) = 2*(q(3)*p(0) - q(1)*p(1) + q(0)*p(1));

      dgamma_dq(0, 2) = 2*(-q(2)*p(0) + q(2)*p(1) + q(0)*p(2));
      dgamma_dq(1, 2) = qtp;
      dgamma_dq(2, 2) = 2*(q(3)*p(1) - q(2)*p(2) - q(0)*p(0));

      dgamma_dq(0, 3) = 2*(-q(3)*p(0) + q(1)*p(2) - q(0)*p(1));
      dgamma_dq(1, 3) = 2*(-q(3)*p(1) + q(2)*p(2) + q(0)*p(0));
      dgamma_dq(2, 3) = qtp;

      dr_dq = K*(psi*dgamma_dq + npsi2*gamma*RowVector4d(2*(q(0)*p(2) - q(2)*p(0) + q(1)*p(1)),
                                                         2*(q(3)*p(0) - q(1)*p(2) + q(0)*p(1)),
                                                         2*(q(3)*p(1) - q(2)*p(2) - q(0)*p(0)),
                                                         2*(q(1)*p(0) + q(2)*p(1) + q(3)*p(2))));

      // Translation
      Map<Matrix<double, 2, 3, RowMajor>> dr_dt(jacobians[0] + 8);
      dr_dt = psi*K + K*gamma*RowVector3d(0, 0, npsi2);
    }
    // Jacobians wrt feature position
    if (jacobians[1]) {
      Map<Matrix<double, 2, 3, RowMajor>> dr_dp(jacobians[1]);
      dr_dp = K*(psi*LR + npsi2*gamma*RowVector3d(q(1)*q(3) - q(0)*q(2),
                                                  q(2)*q(3) + q(0)*q(1),
                                                  q(0)*q(0) + q(3)*q(3) - q(1)*q(1) - q(2)*q(2))); 
    }
  }

  return true;
}
