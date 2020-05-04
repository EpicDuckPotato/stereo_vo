#ifndef CAMERA_INFO_H_
#define CAMERA_INFO_H_

struct CameraInfo {
  // Intrinsics
  double focal;
  double cx;
  double cy;

  // Distortion parameters
  double k1;
  double k2;
  double p1;
  double p2;
};

#endif
