#include "image_processor.hpp"

using namespace Eigen;
using namespace std;

ImageProcessor::ImageProcessor(cv::Mat cam_mat, 
                               shared_ptr<FeatureTracker> tracker,
                               shared_ptr<BundleAdjuster> adjuster,
                               float bline,
                               float min_feature_distance,
                               float parallax_thresh) : camera_matrix(cam_mat),
                                                        feature_tracker(tracker),
                                                        bundle_adjuster(adjuster),
                                                        baseline(bline),
                                                        min_feature_distance(min_feature_distance),
                                                        parallax_thresh(parallax_thresh) {}

void ImageProcessor::process(const StereoPair &stereo_pair) {
  // Detect features in left image
  vector<cv::Point2f> detected_features;

  cv::goodFeaturesToTrack(stereo_pair.left, detected_features, 300, 0.1, min_feature_distance);
  if (detected_features.size() < 4) {
    return;
  }

  shared_ptr<Keyframe> last_keyframe = bundle_adjuster->get_last_keyframe();

  // If we don't have any keyframes, make this the first one
  if (last_keyframe == nullptr) {
    vector<cv::Point3f> features_3d;
    vector<cv::Point2f> valid_features_2d;

    triangulate_stereo(features_3d,
                       valid_features_2d,
                       detected_features,
                       stereo_pair.left,
                       stereo_pair.right,
                       cv::Mat::eye(4, 4, CV_32F));

    shared_ptr<Keyframe> new_keyframe = make_shared<Keyframe>(Vector3f::Zero(),
                                                              Quaternionf::Identity(),
                                                              stereo_pair.left,
                                                              vector<cv::Point2f>(), // no tracked features
                                                              vector<size_t>(), // no tracked features
                                                              valid_features_2d,
                                                              features_3d);

    bundle_adjuster->add_keyframe(new_keyframe);

    feature_tracker->init(stereo_pair.left, new_keyframe->new_features_2d,
                          new_keyframe->new_ids);

    tvec = cv::Mat::zeros(3, 1, CV_32F);
    rvec = cv::Mat::zeros(3, 1, CV_32F);

    return;
  }

  float av_parallax = 0;
  float percent_lost = 0;
  feature_tracker->track_features(av_parallax, percent_lost, stereo_pair.left, true);
  if (av_parallax <= parallax_thresh && percent_lost < 0.4) {
    return;
  }

  vector<cv::Point2f> tracked_features;
  vector<cv::Point3f> tracked_world_points;
  vector<size_t> tracked_ids;

  feature_tracker->get_tracked_features(tracked_features, tracked_ids);
  bundle_adjuster->get_world_points(tracked_world_points, tracked_ids);

  // Use PnP to get pose of world frame with respect to the current camera frame
  cv::Mat inlier_indices;
  solvePnPRansac(tracked_world_points,
                 tracked_features,
                 camera_matrix,
                 cv::Mat::zeros(4, 1, CV_32F),
                 rvec, tvec, true, 100, 8.0, 0.99, inlier_indices);

  size_t num_inliers = inlier_indices.rows;

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);

  Matrix3f rmat_eigen;
  Vector3f tvec_eigen;
  cv::cv2eigen(rmat, rmat_eigen);
  cv::cv2eigen(tvec, tvec_eigen);

  Quaternionf orientation(rmat_eigen);

  // Allocate memory for new keyframe
  shared_ptr<Keyframe> new_keyframe = make_shared<Keyframe>(tvec_eigen,
                                                            orientation,
                                                            stereo_pair.left,
                                                            vector<cv::Point2f>(num_inliers),
                                                            vector<size_t>(num_inliers),
                                                            vector<cv::Point2f>(),
                                                            vector<cv::Point3f>());

  // Populate new keyframe's features with inliers
  for (size_t i = 0; i < num_inliers; i++) {
    size_t idx = inlier_indices.at<int>(i, 0);
    new_keyframe->tracked_ids[i] = tracked_ids[idx];
    new_keyframe->tracked_features_2d[i] = tracked_features[idx];
  }

  // Now we need to triangulate newly detected features. However, many of these
  // features are probably the same ones we're already tracking. Therefore, filter
  // those out, then triangulate
  size_t num_features = detected_features.size();
  vector<cv::Point2f> new_features;
  for (size_t i = 0; i < num_features; i++) {
    bool tracked = false;
    for (size_t j = 0; j < num_inliers; j++) {
      float dx = detected_features[i].x - new_keyframe->tracked_features_2d[j].x;
      float dy = detected_features[i].y - new_keyframe->tracked_features_2d[j].y;
      if (sqrt(dx*dx + dy*dy) < min_feature_distance) {
        tracked = true;
        break;
      }
    }
    if (!tracked) {
      new_features.push_back(detected_features[i]);
    }
  }

  cv::Mat hmat = cv::Mat::eye(4, 4, CV_32F);
  cv::Mat tmp = rmat.t();
  tmp.copyTo(hmat(cv::Rect(0, 0, 3, 3)));
  tmp = -tmp*tvec;
  tmp.copyTo(hmat(cv::Rect(3, 0, 1, 3)));

  // Triangulate 3d points for detected features
  triangulate_stereo(new_keyframe->new_features_3d,
                     new_keyframe->new_features_2d,
                     new_features,
                     stereo_pair.left,
                     stereo_pair.right,
                     hmat);

  bundle_adjuster->add_keyframe(new_keyframe);

  feature_tracker->draw_track();

  vector<cv::Point2f> features_2d_for_tracker;
  vector<size_t> ids_for_tracker;
  features_2d_for_tracker.insert(features_2d_for_tracker.end(),
                        new_keyframe->tracked_features_2d.begin(),
                        new_keyframe->tracked_features_2d.end());
  features_2d_for_tracker.insert(features_2d_for_tracker.end(),
                        new_keyframe->new_features_2d.begin(),
                        new_keyframe->new_features_2d.end());
  ids_for_tracker.insert(ids_for_tracker.end(),
                         new_keyframe->tracked_ids.begin(),
                         new_keyframe->tracked_ids.end());
  ids_for_tracker.insert(ids_for_tracker.end(),
                         new_keyframe->new_ids.begin(),
                         new_keyframe->new_ids.end());
  feature_tracker->init(stereo_pair.left, features_2d_for_tracker, ids_for_tracker);
}

void ImageProcessor::triangulate_stereo(vector<cv::Point3f>& features_3d,
                                        vector<cv::Point2f>& valid_features_2d,
                                        const vector<cv::Point2f>& left_features_2d,
                                        const cv::Mat& left,
                                        const cv::Mat& right,
                                        const cv::Mat& camera_pose) {

  // Get disparity 
  cv::Mat disparity;
  cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(16*3, 21);
  sbm->compute(left, right, disparity);
  disparity.convertTo(disparity, CV_32F, 1.0/16);

  float focal = camera_matrix.at<float>(0, 0);
  float cx = camera_matrix.at<float>(0, 2);
  float cy = camera_matrix.at<float>(1, 2);

  // Reprojection matrix
  cv::Mat Q = cv::Mat::zeros(4, 4, CV_32F);
  Q.at<float>(0, 0) = 1.0/focal;
  Q.at<float>(1, 1) = 1.0/focal;
  Q.at<float>(0, 3) = -cx/focal;
  Q.at<float>(1, 3) = -cy/focal;
  Q.at<float>(2, 3) = 1;
  Q.at<float>(3, 2) = 1.0/(baseline*focal);

  for (vector<cv::Point2f>::const_iterator it = left_features_2d.begin();
       it != left_features_2d.end(); ++it) {
    float disp = disparity.at<float>(it->y, it->x);
    if (disp > 0) {
      valid_features_2d.push_back(*it);
      cv::Mat disparity_point = cv::Mat::zeros(4, 1, CV_32F);
      disparity_point.at<float>(0) = it->x;
      disparity_point.at<float>(1) = it->y;
      disparity_point.at<float>(2) = disp;
      disparity_point.at<float>(3) = 1;

      cv::Mat world_point = camera_pose*Q*disparity_point;
      features_3d.push_back(cv::Point3f(world_point.at<float>(0)/world_point.at<float>(3),
                                        world_point.at<float>(1)/world_point.at<float>(3),
                                        world_point.at<float>(2)/world_point.at<float>(3)));
    }
  }
}
