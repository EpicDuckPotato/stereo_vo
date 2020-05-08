#include "bundle_adjuster.hpp"
#include "reprojection_error.hpp"
#include "reprojection_factor.hpp"
#include "ceres/loss_function.h"

BundleAdjuster::BundleAdjuster(size_t _window_size, CameraInfo info) {
  window_size = _window_size;

  // Set solver options
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.max_solver_time_in_seconds = 0.1;
  options.num_threads = 4;

  camera_info = info;

  ceres::Problem::Options problem_options;
  problem_options.enable_fast_removal = true;
  problem = make_unique<ceres::Problem>(problem_options);
  se3param = make_unique<ceres::ProductParameterization>(new ceres::QuaternionParameterization(),
                                                         new ceres::IdentityParameterization(3));
}

// Remove oldest pose variable from window
void BundleAdjuster::remove_oldest_pose() {
  shared_ptr<PoseVariable> oldest_pose = pose_window.front();
  for (pair<ceres::ResidualBlockId, size_t> observation : oldest_pose->observations) {
    problem->RemoveResidualBlock(observation.first);
    features[observation.second].refcount--;
    
    // Determine whether this feature was observed at any pose in the window. If not,
    // remove it from the feature list.
    if (features[observation.second].refcount == 0) {
      avail_ids.push(observation.second);
    }
  }
  pose_window.pop();
}

void BundleAdjuster::add_keyframe(shared_ptr<Keyframe> keyframe) {
  // How many features in this keyframe did we track from the previous one?
  size_t num_tracked = keyframe->feature_ids.size();

  // How many features in the keyframe total?
  size_t num_features = keyframe->features_2d.size();

  shared_ptr<PoseVariable> pose_var = make_shared<PoseVariable>(); 

  pose_var->pose[0] = keyframe->orientation.w();
  pose_var->pose[1] = keyframe->orientation.x();
  pose_var->pose[2] = keyframe->orientation.y();
  pose_var->pose[3] = keyframe->orientation.z();

  pose_var->pose[4] = keyframe->position(0);
  pose_var->pose[5] = keyframe->position(1);
  pose_var->pose[6] = keyframe->position(2);

  for (size_t i = 0; i < num_features && i < max_features; i++) {
    // Add a residual block for the keyframe's observation of this feature
    cv::Point2f feature = keyframe->features_2d[i];
    ReprojectionFactor *cost_func = new ReprojectionFactor(feature.x, feature.y, camera_info);

    size_t feature_id;
    if (i < num_tracked) {
      // Feature is already in the list
      feature_id = keyframe->feature_ids[i];
    } else {
      // Feature isn't in the list, so add it to the list

      if (avail_ids.size() > 0) {
        // We have empty positions in the list
        feature_id = avail_ids.top(); 
        avail_ids.pop(); 
      } else {
        // We need to expand the list
        features.push_back(Feature());
        feature_id = features.size() - 1;
      }

      features[feature_id].position = new double[3];
      features[feature_id].position[0] = keyframe->features_3d[i].x;
      features[feature_id].position[1] = keyframe->features_3d[i].y;
      features[feature_id].position[2] = keyframe->features_3d[i].z;

      keyframe->feature_ids.push_back(feature_id);
    }
    features[feature_id].refcount++;

    pose_var->observations.push_back(make_pair(problem->AddResidualBlock(cost_func,
                                                                         NULL /* squared loss */,
                                                                         pose_var->pose,
                                                                         features[feature_id].position),
                                                                         feature_id));
  }
  problem->SetParameterization(pose_var->pose, se3param.get());

  keyframe->features_2d.resize(keyframe->feature_ids.size());
  keyframe->features_3d.resize(keyframe->feature_ids.size());

  pose_window.push(pose_var);
  if (pose_window.size() > window_size) {
    remove_oldest_pose();
  }

  last_keyframe = keyframe;
}

void BundleAdjuster::bundle_adjust() { 
  ceres::Solver::Summary summary;
  ceres::Solve(options, problem.get(), &summary);

  shared_ptr<PoseVariable> pose_var = pose_window.back();
  last_keyframe->orientation.w() = pose_var->pose[0];
  last_keyframe->orientation.x() = pose_var->pose[1];
  last_keyframe->orientation.y() = pose_var->pose[2];
  last_keyframe->orientation.z() = pose_var->pose[3];
  last_keyframe->position(0) = pose_var->pose[4];
  last_keyframe->position(1) = pose_var->pose[5];
  last_keyframe->position(2) = pose_var->pose[6];
  
  size_t visible_features = last_keyframe->feature_ids.size();
  for (size_t i = 0; i < visible_features; i++) {
    size_t id = last_keyframe->feature_ids[i];
    last_keyframe->features_3d[i].x = features[id].position[0];
    last_keyframe->features_3d[i].y = features[id].position[1];
    last_keyframe->features_3d[i].z = features[id].position[2];
  }
}
