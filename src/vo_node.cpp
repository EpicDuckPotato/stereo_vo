#include <iostream>
#include <sensor_msgs/Image.h>
#include "ros/ros.h"
#include <Eigen/Dense>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/eigen.hpp"
#include <cv_bridge/cv_bridge.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include "bundle_adjuster.hpp"

using namespace Eigen;
using namespace std;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
                                                        sensor_msgs::Image> ApproxSyncPolicy;

// KITTI sequence 00
static const float focal_length = 7.18856e+02;
static const float cx = 6.071928e+02;
static const float cy = 1.852157e+02;
static const float baseline = 0.537165718864418;

// KITTI sequence 03
/*
static const float focal_length = 7.215377e+02;
static const float cx = 6.095593e+02;
static const float cy = 1.72854e+02;
static const float baseline = 0.537150588250621;
*/

// KITTI sequence 13
/*
static const float focal_length = 7.18856e+02;
static const float cx = 6.071928e+02;
static const float cy = 1.852157e+02;
static const float baseline = 0.537165718864418;
*/

// Realsense d435i
/*
static const float focal_length = 385.7544860839844;
static const float cx = 323.1204833984375;
static const float cy = 236.7432098388672;
static const float baseline = 0.05;
*/

// Realsense r200 (in simulation, the left and right images seem identical?)
/*
static const float focal_length = 554.3826904296875;
static const float cx = 320;
static const float cy = 240;
static const float baseline = 0.07;
*/

// Multisense
/*
static const float focal_length = 476.7030836014194;
static const float cx = 400.5;
static const float cy = 400.5;
static const float baseline = 0.07;
*/

static const float parallax_thresh = 20;
static const float min_feature_distance = 30;

static cv::Mat camera_matrix;

static BundleAdjuster bundle_adjuster(5, {focal_length, cx, cy, 0, 0, 0, 0});

static cv::Mat rvec;
static cv::Mat tvec;

void right_projection_matrix(cv::Mat &right_pmat, const cv::Mat &left_pmat) {
  right_pmat = left_pmat.clone();
  right_pmat.at<float>(0, 3) += focal_length*baseline;
}

// Computes the 3d positions of the features
// REQUIRES: features_3d.size() == 0 && matched_features_2d.size() == 0
void triangulate_stereo(vector<cv::Point3f>& features_3d,
                        vector<cv::Point2f>& matched_features_2d,
                        const vector<cv::Point2f>& left_features_2d,
                        const cv::Mat& left,
                        const cv::Mat& right,
                        const cv::Mat& left_pmat) {

  cv::Mat right_pmat;
  right_projection_matrix(right_pmat, left_pmat);

  cv::Mat disparity;
  cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(16*3, 21);
  sbm->compute(left, right, disparity);

  // Show disparity
  /*
  float minVal, maxVal;
  cv::minMaxLoc(disparity, &minVal, &maxVal);
  cv::Mat disp_u;
  disparity.convertTo(disp_u, CV_8UC1, 255/(maxVal - minVal));
  cv::imshow("disp", disp_u);
  cv::waitKey(0);
  */

  disparity.convertTo(disparity, CV_32F, 1.0/16);

  size_t num_features = left_features_2d.size();

  vector<cv::Point2f> right_features_2d;
  for (size_t i = 0; i < num_features; i++) {
    float disp = disparity.at<float>(left_features_2d[i].y,
                                     left_features_2d[i].x);
    
    if (disp > 0) {
      matched_features_2d.push_back(left_features_2d[i]);
      right_features_2d.push_back(left_features_2d[i] + cv::Point2f(disp, 0));
    }
  }

  num_features = matched_features_2d.size();
  cv::Mat points_homogeneous = cv::Mat::zeros(4, num_features, CV_32F);

  if (num_features > 0) {
    cv::triangulatePoints(left_pmat, right_pmat, matched_features_2d,
                          right_features_2d, points_homogeneous);

    for (size_t i = 0; i < num_features; i++) {
      points_homogeneous.colRange(i, i + 1) /= points_homogeneous.at<float>(3, i);
      features_3d.push_back(cv::Point3f(points_homogeneous.at<float>(0, i),
                                        points_homogeneous.at<float>(1, i),
                                        points_homogeneous.at<float>(2, i)));
    }
  }
}

// As we get image pairs from ROS, we use them for processing only if features
// from the last keyframe have sufficient parallax with this pair's left image.
// If so, use previous keyframe's 3d feature positions to solve for current left
// camera pose via PnP
void handle_images(const sensor_msgs::ImageConstPtr& left_msg,
                   const sensor_msgs::ImageConstPtr& right_msg) { 
  cv_bridge::CvImagePtr left_ptr = cv_bridge::toCvCopy(*left_msg, sensor_msgs::image_encodings::MONO8);
  cv_bridge::CvImagePtr right_ptr = cv_bridge::toCvCopy(*right_msg, sensor_msgs::image_encodings::MONO8);

  // Detect features in left image
  vector<cv::KeyPoint> detected_keypoints;
  vector<cv::Point2f> detected_features;

  // Images from r200 are especially noisy if you don't blur
  /*
  cv::GaussianBlur(left_ptr->image, left_ptr->image, cv::Size(3, 3), 1);
  cv::GaussianBlur(right_ptr->image, right_ptr->image, cv::Size(3, 3), 1);
  */

  /*
  cv::FAST(left_ptr->image, detected_keypoints, 40); // better for KITTI
  //cv::FAST(left_ptr->image, detected_keypoints, 10); // better for d435i in snake bags

  if (detected_keypoints.size() < 4) {
    return;
  }

  cv::KeyPoint::convert(detected_keypoints, detected_features);
  */

  cv::goodFeaturesToTrack(left_ptr->image, detected_features, 300, 0.01, min_feature_distance);
  if (detected_features.size() < 4) {
    return;
  }

  shared_ptr<Keyframe> last_keyframe = bundle_adjuster.get_last_keyframe();

  // If we don't have any keyframes, make this the first one
  if (last_keyframe == nullptr) {
    vector<cv::Point3f> features_3d;
    vector<cv::Point2f> matched_features_2d;

    triangulate_stereo(features_3d,
                       matched_features_2d,
                       detected_features,
                       left_ptr->image,
                       right_ptr->image,
                       camera_matrix);

    shared_ptr<Keyframe> new_keyframe = make_shared<Keyframe>(Vector3f::Zero(),
                                                              Quaternionf::Identity(),
                                                              left_ptr->image,
                                                              matched_features_2d,
                                                              features_3d,
                                                              vector<size_t>());

    bundle_adjuster.add_keyframe(new_keyframe);
    
    tvec = cv::Mat::zeros(3, 1, CV_32F);
    rvec = cv::Mat::zeros(3, 1, CV_32F);

    return;
  }

  // Track features between last keyframe and current frame
  vector<uchar> status;
  vector<float> err;
  vector<cv::Point2f> tracked_features;
  cv::calcOpticalFlowPyrLK(last_keyframe->image, left_ptr->image, last_keyframe->features_2d, tracked_features, status,
                           err, cv::Size(21, 21), 3,
                           cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),
                           0, 1e-4);

  size_t num_features = tracked_features.size();

  // We might've lost some features. Therefore, create vectors of valid features for PnP. Also,
  // if average parallax is less than threshold, don't use this frame as a keyframe
  vector<cv::Point2f> valid_tracked_features;
  vector<cv::Point3f> valid_features_3d;
  vector<size_t> valid_ids; 

  float av_parallax;
  for (size_t i = 0; i < num_features; i++) {
    if (status[i]) {
      valid_tracked_features.push_back(tracked_features[i]);
      valid_features_3d.push_back(last_keyframe->features_3d[i]);
      valid_ids.push_back(last_keyframe->feature_ids[i]);

      float dx = tracked_features[i].x - last_keyframe->features_2d[i].x;
      float dy = tracked_features[i].y - last_keyframe->features_2d[i].y;

      av_parallax += sqrt(dx*dx + dy*dy);
    }
  }
  num_features = valid_tracked_features.size();
  av_parallax /= num_features;

  if (av_parallax <= parallax_thresh) {
    return;
  }

  // Use PnP to get transform between last keyframe and this one
  cv::Mat inlier_indices;
  solvePnPRansac(valid_features_3d,
                 valid_tracked_features,
                 camera_matrix.colRange(0, 3),
                 cv::Mat::zeros(4, 1, CV_32F),
                 rvec, tvec, true, 100, 8.0, 0.99, inlier_indices);

  size_t num_inliers = inlier_indices.rows;

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);

  Matrix3f rmat_eigen;
  Vector3f tvec_eigen;
  cv::cv2eigen(rmat, rmat_eigen);
  cv::cv2eigen(tvec, tvec_eigen);

  // Get new keyframe's pose
  Quaternionf orientation(rmat_eigen);

  // Allocate memory for new keyframe
  shared_ptr<Keyframe> new_keyframe = make_shared<Keyframe>(tvec_eigen,
                                                            orientation,
                                                            left_ptr->image,
                                                            vector<cv::Point2f>(num_inliers),
                                                            vector<cv::Point3f>(num_inliers),
                                                            vector<size_t>(num_inliers));

  // Populate new keyframe's features with inliers
  for (size_t i = 0; i < num_inliers; i++) {
    size_t idx = inlier_indices.at<int>(i, 0);
    new_keyframe->feature_ids[i] = valid_ids[idx];
    new_keyframe->features_2d[i] = valid_tracked_features[idx];
    new_keyframe->features_3d[i] = valid_features_3d[idx];
  }

  // Now we need to triangulate newly detected features. However, many of these
  // features are probably the same ones we're already tracking. Therefore, filter
  // those out, then triangulate
  num_features = detected_features.size();
  vector<cv::Point2f> new_features;
  for (size_t i = 0; i < num_features; i++) {
    bool tracked = false;
    for (size_t j = 0; j < num_inliers; j++) {
      float dx = detected_features[i].x - new_keyframe->features_2d[j].x;
      float dy = detected_features[i].y - new_keyframe->features_2d[j].y;
      if (sqrt(dx*dx + dy*dy) < min_feature_distance) {
        tracked = true;
        break;
      }
    }
    if (!tracked) {
      new_features.push_back(detected_features[i]);
    }
  }

  vector<cv::Point3f> new_features_3d;
  vector<cv::Point2f> matched_new_features;

  cv::Mat hmat = cv::Mat::eye(4, 4, CV_32F);
  rmat.copyTo(hmat(cv::Rect(0, 0, 3, 3)));
  tvec.copyTo(hmat(cv::Rect(3, 0, 1, 3)));

  // Triangulate 3d points for detected features
  triangulate_stereo(new_features_3d,
                     matched_new_features,
                     new_features,
                     left_ptr->image,
                     right_ptr->image,
                     camera_matrix*hmat);

  new_keyframe->features_2d.insert(new_keyframe->features_2d.end(), 
                                   make_move_iterator(matched_new_features.begin()),
                                   make_move_iterator(matched_new_features.end()));

  new_keyframe->features_3d.insert(new_keyframe->features_3d.end(), 
                                   make_move_iterator(new_features_3d.begin()),
                                   make_move_iterator(new_features_3d.end()));

  bundle_adjuster.add_keyframe(new_keyframe);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "vo_node");
  ros::NodeHandle n("~");

  // KITTI topic
  message_filters::Subscriber<sensor_msgs::Image> left_sub(n, "/leftImage", 10);
  message_filters::Subscriber<sensor_msgs::Image> right_sub(n, "/rightImage", 10);

  // d435i topic
  /*
  message_filters::Subscriber<sensor_msgs::Image> left_sub(n, "/camera/infra1/image_rect_raw", 3);
  message_filters::Subscriber<sensor_msgs::Image> right_sub(n, "/camera/infra2/image_rect_raw", 3);
  */

  // r200 topic
  /*
  message_filters::Subscriber<sensor_msgs::Image> left_sub(n, "/snake_r200/camera/ir/image_raw", 3);
  message_filters::Subscriber<sensor_msgs::Image> right_sub(n, "/snake_r200/camera/ir2/image_raw", 3);
  */

  // multisense topic
  /*
  message_filters::Subscriber<sensor_msgs::Image> left_sub(n, "/multisense_sl/camera/left/image_raw", 3);
  message_filters::Subscriber<sensor_msgs::Image> right_sub(n, "/multisense_sl/camera/right/image_raw", 3);
  */

  camera_matrix = cv::Mat::eye(3, 4, CV_32F);
  camera_matrix.at<float>(0, 0) = focal_length;
  camera_matrix.at<float>(0, 2) = cx;
  camera_matrix.at<float>(1, 1) = focal_length;
  camera_matrix.at<float>(1, 2) = cy;

  message_filters::Synchronizer<ApproxSyncPolicy> sync(ApproxSyncPolicy(10), left_sub, right_sub);
  sync.registerCallback(boost::bind(&handle_images, _1, _2));

  tf2_ros::TransformBroadcaster br;

  //ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("features", 1);
  ros::Publisher path_pub = n.advertise<nav_msgs::Path>("/vo/path", 1);
  nav_msgs::Path path;
  path.header.frame_id = "world";

  ros::Rate r(20);
  while (ros::ok()) {
    if (bundle_adjuster.get_last_keyframe() != nullptr) {
      shared_ptr<Keyframe> keyframe = bundle_adjuster.get_last_keyframe();
      Quaternionf orientation = keyframe->orientation.conjugate();
      Vector3f position = orientation*(-keyframe->position);

      geometry_msgs::TransformStamped pose;
      pose.transform.rotation.w = orientation.w();
      pose.transform.rotation.x = orientation.x();
      pose.transform.rotation.y = orientation.y();
      pose.transform.rotation.z = orientation.z();

      pose.transform.translation.x = position(0);
      pose.transform.translation.y = position(1);
      pose.transform.translation.z = position(2);

      pose.header.stamp = ros::Time::now();
      pose.header.frame_id = "world";
      pose.child_frame_id = "vo_pose";

      br.sendTransform(pose);

      geometry_msgs::PoseStamped gpose;
      gpose.pose.orientation.w = orientation.w();
      gpose.pose.orientation.x = orientation.x();
      gpose.pose.orientation.y = orientation.y();
      gpose.pose.orientation.z = orientation.z();

      gpose.pose.position.x = position(0);
      gpose.pose.position.y = position(1);
      gpose.pose.position.z = position(2);

      gpose.header.stamp = pose.header.stamp;
      gpose.header.frame_id = "vo_pose";

      path.poses.push_back(gpose);
      path.header.stamp = gpose.header.stamp;
      path_pub.publish(path);

      // For displaying features in rviz
      /*
      visualization_msgs::Marker marker;
      marker.header.stamp = pose.header.stamp;
      marker.header.frame_id = "world";
      marker.ns = "vo_node";
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position.x = 0;
      marker.pose.position.y = 0;
      marker.pose.position.z = 0;
      marker.pose.orientation.w = 1;
      marker.pose.orientation.x = 0;
      marker.pose.orientation.y = 0;
      marker.pose.orientation.z = 0;
      marker.type = visualization_msgs::Marker::POINTS;
      marker.scale.x = 0.2;
      marker.scale.y = 0.2;
      marker.color.g = 1.0f;
      marker.color.a = 1.0;

      size_t num_features = keyframe->features_3d.size();
      for (size_t i = 0; i < num_features; ++i) {
        geometry_msgs::Point p;
        p.x = keyframe->features_3d[i].x;
        p.y = keyframe->features_3d[i].y;
        p.z = keyframe->features_3d[i].z;

        marker.points.push_back(p);
      }
 
      marker_pub.publish(marker);
      */

      bundle_adjuster.bundle_adjust();
    }

    r.sleep();
    ros::spinOnce();
  }

  return 0;
}
