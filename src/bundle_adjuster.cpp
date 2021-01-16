#include "bundle_adjuster.hpp"
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

  new_frame_added = false;
}

BundleAdjuster::~BundleAdjuster() {
  // Free memory for feature positions
  for (Feature feature : features) {
    if (feature.position != nullptr) {
      delete [] feature.position;
    }
  }

  // Free memory for pose variables
  while (pose_window.size() != 0) {
    delete [] pose_window.front()->pose;
    pose_window.pop();
  }
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
      problem->RemoveParameterBlock(features[observation.second].position);
      delete [] features[observation.second].position;
    }
  }
  problem->RemoveParameterBlock(oldest_pose->pose);
  delete [] oldest_pose->pose;
  pose_window.pop();
}

void BundleAdjuster::add_keyframe(shared_ptr<Keyframe> keyframe) {
  shared_ptr<PoseVariable> pose_var = make_shared<PoseVariable>(); 

  pose_var->pose[0] = keyframe->orientation.w();
  pose_var->pose[1] = keyframe->orientation.x();
  pose_var->pose[2] = keyframe->orientation.y();
  pose_var->pose[3] = keyframe->orientation.z();

  pose_var->pose[4] = keyframe->position(0);
  pose_var->pose[5] = keyframe->position(1);
  pose_var->pose[6] = keyframe->position(2);

  size_t num_tracked = keyframe->tracked_ids.size();
  for (size_t i = 0; i < num_tracked; ++i) {
    cv::Point2f feature = keyframe->tracked_features_2d[i];
    ReprojectionFactor *cost_func = new ReprojectionFactor(feature.x, feature.y, camera_info);
    size_t feature_id = keyframe->tracked_ids[i];
    features[feature_id].refcount++;
    pose_var->observations.push_back(make_pair(problem->AddResidualBlock(cost_func,
                                                                         NULL /* squared loss */,
                                                                         pose_var->pose,
                                                                         features[feature_id].position),
                                                                         feature_id));
  }

  size_t max_new = max_features - num_tracked;
  if (keyframe->new_features_2d.size() > max_new) {
    keyframe->new_features_2d.resize(max_new);
    keyframe->new_features_3d.resize(max_new);
    keyframe->new_ids.resize(max_new);
  }

  size_t num_new = keyframe->new_features_2d.size();
  for (size_t i = 0; i < num_new; ++i) {
    // Add a residual block for the keyframe's observation of this feature
    cv::Point2f feature = keyframe->new_features_2d[i];
    ReprojectionFactor *cost_func = new ReprojectionFactor(feature.x, feature.y, camera_info);

    size_t feature_id;
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
    features[feature_id].position[0] = keyframe->new_features_3d[i].x;
    features[feature_id].position[1] = keyframe->new_features_3d[i].y;
    features[feature_id].position[2] = keyframe->new_features_3d[i].z;
    features[feature_id].refcount = 1;

    keyframe->new_ids.push_back(feature_id);
    features[feature_id].refcount++;
    pose_var->observations.push_back(make_pair(problem->AddResidualBlock(cost_func,
                                                                         NULL /* squared loss */,
                                                                         pose_var->pose,
                                                                         features[feature_id].position),
                                                                         feature_id));
  }
  problem->SetParameterization(pose_var->pose, se3param.get());

  pose_window.push(pose_var);
  if (pose_window.size() > window_size) {
    remove_oldest_pose();
  }

  problem->SetParameterBlockConstant(pose_window.front()->pose);

  last_keyframe = keyframe;

  new_frame_added = true;
}

void BundleAdjuster::bundle_adjust() { 
  if (new_frame_added) {
    ceres::Solver::Summary summary;
    ceres::Solve(options, problem.get(), &summary);

    if (summary.termination_type == ceres::TerminationType::NO_CONVERGENCE) {
      std::cout << "Solver didn't converge!" << std::endl;
    }

    shared_ptr<PoseVariable> pose_var = pose_window.back();
    last_keyframe->orientation.w() = pose_var->pose[0];
    last_keyframe->orientation.x() = pose_var->pose[1];
    last_keyframe->orientation.y() = pose_var->pose[2];
    last_keyframe->orientation.z() = pose_var->pose[3];
    last_keyframe->position(0) = pose_var->pose[4];
    last_keyframe->position(1) = pose_var->pose[5];
    last_keyframe->position(2) = pose_var->pose[6];
    
    new_frame_added = false;
  }
}

void BundleAdjuster::get_world_points(vector<cv::Point3f> &world_points, const vector<size_t> &ids) {
  for (vector<size_t>::const_iterator it = ids.begin(); it != ids.end(); ++it) {
    world_points.push_back(cv::Point3f(features[*it].position[0], features[*it].position[1], features[*it].position[2]));
  }
}
