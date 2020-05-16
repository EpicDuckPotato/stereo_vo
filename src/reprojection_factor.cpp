#include "reprojection_factor.hpp"

ReprojectionFactor::ReprojectionFactor(double ox, double oy, CameraInfo info) {
  obs = Vector2d(ox, oy);
  K << info.focal, 0         , info.cx,
       0         , info.focal, info.cy;
  camera_info = info;
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

  Vector3d gamma = LR*p/q.squaredNorm() + t;

  double psi = 1/gamma(2);

  // Calculate residual
  resid = K*psi*gamma - obs;

  double q1 = parameters[0][0];
  double q2 = parameters[0][1];
  double q3 = parameters[0][2];
  double q4 = parameters[0][3];
  double t1 = parameters[0][4];
  double t2 = parameters[0][5];
  double t3 = parameters[0][6];
  double p1 = parameters[1][0];
  double p2 = parameters[1][1];
  double p3 = parameters[1][2];

  // Residual expressions from MATLAB symbolic computation. Didn't seem necessary to use them
  /*
  residuals[0] = camera_info.cx-obs(0)+(camera_info.focal*(t1+q1*(p1*q1-p2*q4+p3*q3)+q3*(-p1*q3+p2*q2+p3*q1)+q2*(p1*q2+p2*q3+p3*q4)-q4*(p2*q1+p1*q4-p3*q2)))/(t3+q1*(-p1*q3+p2*q2+p3*q1)+q2*(p2*q1+p1*q4-p3*q2)-q3*(p1*q1-p2*q4+p3*q3)+q4*(p1*q2+p2*q3+p3*q4));
  residuals[1] = camera_info.cy-obs(1)+(camera_info.focal*(t2+q1*(p2*q1+p1*q4-p3*q2)-q2*(-p1*q3+p2*q2+p3*q1)+q3*(p1*q2+p2*q3+p3*q4)+q4*(p1*q1-p2*q4+p3*q3)))/(t3+q1*(-p1*q3+p2*q2+p3*q1)+q2*(p2*q1+p1*q4-p3*q2)-q3*(p1*q1-p2*q4+p3*q3)+q4*(p1*q2+p2*q3+p3*q4));
  */

  // Compute jacobians (from MATLAB symbolic differentiation)
  if (jacobians) {
    if (jacobians[0]) {
      Map<Matrix<double, 2, 7, RowMajor>> dr_dpose(jacobians[0]);
      dr_dpose.setZero();

      jacobians[0][0] = (camera_info.focal*((p1*q1*2.0-p2*q4*2.0+p3*q3*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)-q1*(p1*(q1*q1+q2*q2-q3*q3-q4*q4)-p2*(q1*q4*2.0-q2*q3*2.0)+p3*(q1*q3*2.0+q2*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0))/(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))-camera_info.focal*(t1+(p1*(q1*q1+q2*q2-q3*q3-q4*q4)-p2*(q1*q4*2.0-q2*q3*2.0)+p3*(q1*q3*2.0+q2*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0)*((p1*q3*-2.0+p2*q2*2.0+p3*q1*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)-q1*(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0);
      jacobians[0][1] = (camera_info.focal*((p1*q2*2.0+p2*q3*2.0+p3*q4*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)-q2*(p1*(q1*q1+q2*q2-q3*q3-q4*q4)-p2*(q1*q4*2.0-q2*q3*2.0)+p3*(q1*q3*2.0+q2*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0))/(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))-camera_info.focal*(t1+(p1*(q1*q1+q2*q2-q3*q3-q4*q4)-p2*(q1*q4*2.0-q2*q3*2.0)+p3*(q1*q3*2.0+q2*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0)*((p2*q1*2.0+p1*q4*2.0-p3*q2*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)-q2*(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0);
      jacobians[0][2] = (camera_info.focal*((p1*q3*-2.0+p2*q2*2.0+p3*q1*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)-q3*(p1*(q1*q1+q2*q2-q3*q3-q4*q4)-p2*(q1*q4*2.0-q2*q3*2.0)+p3*(q1*q3*2.0+q2*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0))/(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))+camera_info.focal*(t1+(p1*(q1*q1+q2*q2-q3*q3-q4*q4)-p2*(q1*q4*2.0-q2*q3*2.0)+p3*(q1*q3*2.0+q2*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0)*((p1*q1*2.0-p2*q4*2.0+p3*q3*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)+q3*(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0);
      jacobians[0][3] = -(camera_info.focal*((p2*q1*2.0+p1*q4*2.0-p3*q2*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)+q4*(p1*(q1*q1+q2*q2-q3*q3-q4*q4)-p2*(q1*q4*2.0-q2*q3*2.0)+p3*(q1*q3*2.0+q2*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0))/(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))-camera_info.focal*(t1+(p1*(q1*q1+q2*q2-q3*q3-q4*q4)-p2*(q1*q4*2.0-q2*q3*2.0)+p3*(q1*q3*2.0+q2*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0)*((p1*q2*2.0+p2*q3*2.0+p3*q4*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)-q4*(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0);
      jacobians[0][7] = (camera_info.focal*((p2*q1*2.0+p1*q4*2.0-p3*q2*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)-q1*(p2*(q1*q1-q2*q2+q3*q3-q4*q4)+p1*(q1*q4*2.0+q2*q3*2.0)-p3*(q1*q2*2.0-q3*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0))/(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))-camera_info.focal*(t2+(p2*(q1*q1-q2*q2+q3*q3-q4*q4)+p1*(q1*q4*2.0+q2*q3*2.0)-p3*(q1*q2*2.0-q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0)*((p1*q3*-2.0+p2*q2*2.0+p3*q1*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)-q1*(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0);
      jacobians[0][8] = -(camera_info.focal*((p1*q3*-2.0+p2*q2*2.0+p3*q1*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)+q2*(p2*(q1*q1-q2*q2+q3*q3-q4*q4)+p1*(q1*q4*2.0+q2*q3*2.0)-p3*(q1*q2*2.0-q3*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0))/(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))-camera_info.focal*(t2+(p2*(q1*q1-q2*q2+q3*q3-q4*q4)+p1*(q1*q4*2.0+q2*q3*2.0)-p3*(q1*q2*2.0-q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0)*((p2*q1*2.0+p1*q4*2.0-p3*q2*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)-q2*(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0);
      jacobians[0][9] = (camera_info.focal*((p1*q2*2.0+p2*q3*2.0+p3*q4*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)-q3*(p2*(q1*q1-q2*q2+q3*q3-q4*q4)+p1*(q1*q4*2.0+q2*q3*2.0)-p3*(q1*q2*2.0-q3*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0))/(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))+camera_info.focal*(t2+(p2*(q1*q1-q2*q2+q3*q3-q4*q4)+p1*(q1*q4*2.0+q2*q3*2.0)-p3*(q1*q2*2.0-q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0)*((p1*q1*2.0-p2*q4*2.0+p3*q3*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)+q3*(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0);
      jacobians[0][10] = (camera_info.focal*((p1*q1*2.0-p2*q4*2.0+p3*q3*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)-q4*(p2*(q1*q1-q2*q2+q3*q3-q4*q4)+p1*(q1*q4*2.0+q2*q3*2.0)-p3*(q1*q2*2.0-q3*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0))/(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))-camera_info.focal*(t2+(p2*(q1*q1-q2*q2+q3*q3-q4*q4)+p1*(q1*q4*2.0+q2*q3*2.0)-p3*(q1*q2*2.0-q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0)*((p1*q2*2.0+p2*q3*2.0+p3*q4*2.0)/(q1*q1+q2*q2+q3*q3+q4*q4)-q4*(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))*1.0/pow(q1*q1+q2*q2+q3*q3+q4*q4,2.0)*2.0);

      jacobians[0][4] = camera_info.focal/(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4));
      jacobians[0][6] = -camera_info.focal*(t1+(p1*(q1*q1+q2*q2-q3*q3-q4*q4)-p2*(q1*q4*2.0-q2*q3*2.0)+p3*(q1*q3*2.0+q2*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0);
      jacobians[0][12] = camera_info.focal/(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4));
      jacobians[0][13] = -camera_info.focal*(t2+(p2*(q1*q1-q2*q2+q3*q3-q4*q4)+p1*(q1*q4*2.0+q2*q3*2.0)-p3*(q1*q2*2.0-q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0);
    }
    if (jacobians[1]) {
      jacobians[1][0] = (camera_info.focal*(q1*q1+q2*q2-q3*q3-q4*q4))/((t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*(q1*q1+q2*q2+q3*q3+q4*q4))+(camera_info.focal*(t1+(p1*(q1*q1+q2*q2-q3*q3-q4*q4)-p2*(q1*q4*2.0-q2*q3*2.0)+p3*(q1*q3*2.0+q2*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0)*(q1*q3*2.0-q2*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4);
      jacobians[1][1] = -(camera_info.focal*(q1*q4*2.0-q2*q3*2.0))/((t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*(q1*q1+q2*q2+q3*q3+q4*q4))-(camera_info.focal*(t1+(p1*(q1*q1+q2*q2-q3*q3-q4*q4)-p2*(q1*q4*2.0-q2*q3*2.0)+p3*(q1*q3*2.0+q2*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0)*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4);
      jacobians[1][2] = (camera_info.focal*(q1*q3*2.0+q2*q4*2.0))/((t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*(q1*q1+q2*q2+q3*q3+q4*q4))-(camera_info.focal*(t1+(p1*(q1*q1+q2*q2-q3*q3-q4*q4)-p2*(q1*q4*2.0-q2*q3*2.0)+p3*(q1*q3*2.0+q2*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0)*(q1*q1-q2*q2-q3*q3+q4*q4))/(q1*q1+q2*q2+q3*q3+q4*q4);
      jacobians[1][3] = (camera_info.focal*(q1*q4*2.0+q2*q3*2.0))/((t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*(q1*q1+q2*q2+q3*q3+q4*q4))+(camera_info.focal*(t2+(p2*(q1*q1-q2*q2+q3*q3-q4*q4)+p1*(q1*q4*2.0+q2*q3*2.0)-p3*(q1*q2*2.0-q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0)*(q1*q3*2.0-q2*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4);
      jacobians[1][4] = (camera_info.focal*(q1*q1-q2*q2+q3*q3-q4*q4))/((t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*(q1*q1+q2*q2+q3*q3+q4*q4))-(camera_info.focal*(t2+(p2*(q1*q1-q2*q2+q3*q3-q4*q4)+p1*(q1*q4*2.0+q2*q3*2.0)-p3*(q1*q2*2.0-q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0)*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4);
      jacobians[1][5] = -(camera_info.focal*(q1*q2*2.0-q3*q4*2.0))/((t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*(q1*q1+q2*q2+q3*q3+q4*q4))-(camera_info.focal*(t2+(p2*(q1*q1-q2*q2+q3*q3-q4*q4)+p1*(q1*q4*2.0+q2*q3*2.0)-p3*(q1*q2*2.0-q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4))*1.0/pow(t3+(p3*(q1*q1-q2*q2-q3*q3+q4*q4)-p1*(q1*q3*2.0-q2*q4*2.0)+p2*(q1*q2*2.0+q3*q4*2.0))/(q1*q1+q2*q2+q3*q3+q4*q4),2.0)*(q1*q1-q2*q2-q3*q3+q4*q4))/(q1*q1+q2*q2+q3*q3+q4*q4);
    }
  }

  return true;
}
