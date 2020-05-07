#include <ceres/ceres.h>
#include <ceres/rotation.h> 
#include "camera_info.hpp"

struct ReprojectionError {
  ReprojectionError(double observed_x, double observed_y, CameraInfo info)
      : observed_x(observed_x), observed_y(observed_y), camera_info(info) {}

  template <typename T>
  bool operator()(const T* const pose, // wxyz (orientation), xyz (position)
                  const T* const point,
                  T* residuals) const {
    T p[3];
    ceres::QuaternionRotatePoint(pose, point, p);

    p[0] += pose[4];
    p[1] += pose[5];
    p[2] += pose[6];

    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    // Apply second and fourth order radial and tangential distortion.
    /*
    T x2 = xp*xp;
    T y2 = yp*yp;
    T r2 = x2 + y2;
    T cross = 2.0 * T(camera_info.p2) * xp * yp;

    T radial_distortion = T(1.0) + r2 * (T(camera_info.k1) + T(camera_info.k2) * r2);

    T predicted_x = T(camera_info.focal) * (radial_distortion * xp +
                                            T(camera_info.p1) * (r2 + 2.0*x2) +
                                            cross) + camera_info.cx;
    T predicted_y = T(camera_info.focal) * (radial_distortion * yp +
                                            T(camera_info.p1) * (r2 + 2.0*y2) +
                                            cross) + camera_info.cy;
    */
    T predicted_x = T(camera_info.focal) * xp + camera_info.cx;
    T predicted_y = T(camera_info.focal) * yp + camera_info.cy;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(double observed_x, double observed_y, CameraInfo info) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 7, 3>(
                new ReprojectionError(observed_x, observed_y, info)));
  }

  double observed_x;
  double observed_y;
  CameraInfo camera_info;
};
