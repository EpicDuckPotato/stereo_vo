#include "bundle_adjuster.hpp"
#include <gtsam/slam/ProjectionFactor.h>

BundleAdjuster::BundleAdjuster(double lag, CameraInfo info) : smoother(lag) {
  Pose3 priorMean; // Initialize first pose at the origin
  cout << "HERE" << endl;
  noiseModel::Isotropic::shared_ptr priorNoise = noiseModel::Isotropic::Sigma(6, 0.05);
  cout << "HERE" << endl;
  Key priorKey = 0;
  newFactors.addPrior(priorKey, priorMean, priorNoise);
  cout << "HERE" << endl;
  newValues.insert(priorKey, priorMean); 
  cout << "HERE" << endl;
  newTimestamps[priorKey] = 0.0; 

  K = gtsam::make_shared<Cal3Unified>(info.focal, info.focal, 0, info.cx, info.cy,
                                      info.k1, info.k2, info.p1, info.p2);

  next_key = 1;
  new_frame_added = false;
  any_frames = false;
}

void BundleAdjuster::add_keyframe(Keyframe &keyframe) {
  // TODO: I'm making this conversion because it seems like gtsam uses pose of camera with respect
  // to world rather than other way around, but I need to verify this
  Rot3 rot = Rot3::Quaternion(keyframe.orientation.w(),
                              -keyframe.orientation.x(),
                              -keyframe.orientation.y(),
                              -keyframe.orientation.z());

  Vector3f pos_cam_wrt_world = keyframe.orientation.conjugate()*(-keyframe.position);
  Vector3 position(pos_cam_wrt_world(0),
                   pos_cam_wrt_world(1),
                   pos_cam_wrt_world(2));
  Pose3 pose(rot, position);
  pose_key = next_key++;
  newTimestamps[pose_key] = keyframe.timestamp;
  newValues.insert(pose_key, pose);

  size_t num_tracked = keyframe.tracked_keys.size();

  noiseModel::Isotropic::shared_ptr measurementNoise = noiseModel::Isotropic::Sigma(2, 1.0);
  for (size_t i = 0; i < num_tracked; ++i) {
    Point2 feature(keyframe.tracked_features_2d[i].x, keyframe.tracked_features_2d[i].y);
    Key feature_key = keyframe.tracked_keys[i];
    newFactors.push_back(GenericProjectionFactor<Pose3, Point3, Cal3Unified>(feature, measurementNoise,
                                                                             pose_key, feature_key, K));
    newTimestamps[feature_key] = keyframe.timestamp;
  }

  size_t max_new = max_features - num_tracked;
  if (keyframe.new_features_2d.size() > max_new) {
    keyframe.new_features_2d.resize(max_new);
    keyframe.new_features_3d.resize(max_new);
    keyframe.new_keys.resize(max_new);
  }

  size_t num_new = keyframe.new_features_2d.size();
  for (size_t i = 0; i < num_new; ++i) {
    Point2 feature(keyframe.new_features_2d[i].x, keyframe.new_features_2d[i].y);
    Key feature_key = next_key++;
    keyframe.new_keys.push_back(feature_key);
    newFactors.push_back(GenericProjectionFactor<Pose3, Point3, Cal3Unified>(feature, measurementNoise,
                                                                             pose_key, feature_key, K));
    Point3 feature_3d(keyframe.new_features_3d[i].x,
                      keyframe.new_features_3d[i].y,
                      keyframe.new_features_3d[i].z);
    newTimestamps[feature_key] = keyframe.timestamp;
    newValues.insert(feature_key, feature_3d);
  }

  new_frame_added = true;
}

void BundleAdjuster::bundle_adjust() { 
  if (new_frame_added) {
    // TODO: update multiple times?
    smoother.update(newFactors, newValues, newTimestamps);

    Pose3 pose = smoother.calculateEstimate<Pose3>(pose_key);
    gtsam::Quaternion rot = pose.rotation().toQuaternion();

    // Clear contains for the next iteration
    newTimestamps.clear();
    newValues.clear();
    newFactors.resize(0);

    last_position = pose.translation();
    last_orientation.w() = rot.w();
    last_orientation.x() = rot.x();
    last_orientation.y() = rot.y();
    last_orientation.z() = rot.z();
    
    new_frame_added = false;
  }
}

void BundleAdjuster::get_last_pose(Vector3d &position, Quaterniond &orientation) {
  position = last_position;
  orientation = last_orientation;
}

void BundleAdjuster::get_world_points(vector<cv::Point3f> &world_points, const vector<Key> &keys) {
  for (vector<Key>::const_iterator it = keys.begin(); it != keys.end(); ++it) {
    Point3 point = smoother.calculateEstimate<Point3>(*it);
    world_points.push_back(cv::Point3f(point(0), point(1), point(2)));
  }
}
