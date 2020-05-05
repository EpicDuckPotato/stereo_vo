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
#include <memory>

using namespace Eigen;
using namespace std;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
                                                        sensor_msgs::Image> ApproxSyncPolicy;

// Keyframe stored with its associated camera pose, 2d features, and 3d features
struct Keyframe {
  Vector3f position; // variable in bundle adjustment
  Quaternionf orientation; // variable in bundle adjustment
  cv::Mat image; // for optical flow
  vector<cv::Point2f> features_2d; // for optical flow
  vector<cv::Point3f> features_3d; // for PnP

  Keyframe(Vector3f position, Quaternionf orientation, cv::Mat image,
           vector<cv::Point2f> features_2d, vector<cv::Point3f> features_3d) :
           position(position), orientation(orientation), image(image),
           features_2d(features_2d), features_3d(features_3d) {}
};

static cv::Mat left_pmat;
static cv::Mat right_pmat;

static cv::Mat rvec;
static cv::Mat tvec;

// KITTI sequence 00
/*
static const double focal_length = 7.18856e+02;
static const double cx = 6.071928e+02;
static const double cy = 1.852157e+02;
static const double stereo_sep = 0.537165718864418;
*/

// KITTI sequence 03
/*
static const double focal_length = 7.215377000000e+02;
static const double cx = 6.095593000000e+02;
static const double cy = 1.728540000000e+02;
static const double stereo_sep = 0.537150588250621;
*/

// KITTI sequence 13
/*
static const double focal_length = 7.188560000000e+02;
static const double cx = 6.071928000000e+02;
static const double cy = 1.852157000000e+02;
static const double stereo_sep = 0.537165718864418;
*/

// Realsense d435i
/*
static const double focal_length = 385.7544860839844;
static const double cx = 323.1204833984375;
static const double cy = 236.7432098388672;
static const double stereo_sep = 0.05;
*/

// Realsense r200 (in simulation, the left and right images seem identical?)
/*
static const double focal_length = 554.3826904296875;
static const double cx = 320;
static const double cy = 240;
static const double stereo_sep = 0.07;
*/

// Multisense
static const double focal_length = 476.7030836014194;
static const double cx = 400.5;
static const double cy = 400.5;
static const double stereo_sep = 0.07;

static shared_ptr<Keyframe> last_keyframe;
static bool got_first_image = false;

// Computes the 3d positions of the features
// REQUIRES: features_3d.size() == features_2d.size()
void triangulate_stereo(vector<cv::Point3f>& features_3d,
                        const vector<cv::Point2f>& left_features_2d,
                        const cv::Mat& left,
                        const cv::Mat& right) {

  cv::Mat disparity;
  cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(16*3, 21);
  sbm->compute(left, right, disparity);

  // Show disparity
  /*
  double minVal, maxVal;
  cv::minMaxLoc(disparity, &minVal, &maxVal);
  cv::Mat disp_u;
  disparity.convertTo(disp_u, CV_8UC1, 255/(maxVal - minVal));
  cv::imshow("disp", disp_u);
  cv::waitKey(0);
  */

  disparity.convertTo(disparity, CV_32F, 1.0/16);

  size_t num_features = left_features_2d.size();
  vector<cv::Point2f> right_features_2d(left_features_2d);
  for (size_t i = 0; i < num_features; i++) {
    right_features_2d[i].x += disparity.at<float>(left_features_2d[i].y,
                                                  left_features_2d[i].x);
  }

  cv::Mat points_homogeneous = cv::Mat::zeros(4, num_features, CV_32F);

  cv::triangulatePoints(left_pmat, right_pmat, left_features_2d, right_features_2d, points_homogeneous);

  for (size_t i = 0; i < num_features; i++) {
    points_homogeneous.colRange(i, i + 1) /= points_homogeneous.at<float>(3, i);
    features_3d[i].x = points_homogeneous.at<float>(0, i);
    features_3d[i].y = points_homogeneous.at<float>(1, i);
    features_3d[i].z = points_homogeneous.at<float>(2, i);
  }
}

// As we get image pairs from ROS, we use them for processing only if features
// from the last keyframe have sufficient parallax with this pair's left image.
// If so, use previous keyframe's 3d feature positions to solve for current left
// camera pose via PnP. Then get 2d and 3d feature positions for current pair
// and push into queue
void handle_images(const sensor_msgs::ImageConstPtr& left_msg,
                   const sensor_msgs::ImageConstPtr& right_msg) { 
  cv_bridge::CvImagePtr left_ptr = cv_bridge::toCvCopy(*left_msg, sensor_msgs::image_encodings::MONO8);
  cv_bridge::CvImagePtr right_ptr = cv_bridge::toCvCopy(*right_msg, sensor_msgs::image_encodings::MONO8);

  // Detect features in left image
  vector<cv::KeyPoint> left_keypoints;
  vector<cv::Point2f> left_features_2d;

  // Images in Gazebo are especially noisy if you don't blur
  /*
  cv::GaussianBlur(left_ptr->image, left_ptr->image, cv::Size(3, 3), 1);
  cv::GaussianBlur(right_ptr->image, right_ptr->image, cv::Size(3, 3), 1);
  */
  cv::FAST(left_ptr->image, left_keypoints, 40); // better for KITTI
  //cv::FAST(left_ptr->image, left_keypoints, 10); // better for d435i in snake bags

  if (left_keypoints.size() < 4) {
    return;
  }
  cv::KeyPoint::convert(left_keypoints, left_features_2d);

  // If you want to see features
  /*
  for (size_t i = 0; i < left_features_2d.size(); i++) {
    cv::circle(left_ptr->image, left_features_2d[i], 3, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_8);
  }
  cv::imshow("feat", left_ptr->image);
  cv::waitKey(0);
  */

  // If we don't have any keyframes, make this the first one
  if (!got_first_image) {
    got_first_image = true;
    vector<cv::Point3f> features_3d(left_features_2d.size());
    triangulate_stereo(features_3d, left_features_2d, left_ptr->image, right_ptr->image);

    last_keyframe = make_shared<Keyframe>(Vector3f::Zero(),
                                          Quaternionf::Identity(),
                                          left_ptr->image,
                                          left_features_2d,
                                          features_3d);

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

  // If average parallax is less than 10 pixels, don't use this frame as a keyframe
  float av_parallax;
  size_t num_features = tracked_features.size();
  for (size_t i = 0; i < num_features; i++) {
    float dx = tracked_features[i].x - last_keyframe->features_2d[i].x;
    float dy = tracked_features[i].y - last_keyframe->features_2d[i].y;

    av_parallax += sqrt(dx*dx + dy*dy);
  }
  av_parallax /= num_features;

  if (av_parallax <= 20) {
    return;
  }

  // Allocate memory for new keyframe
  num_features = left_features_2d.size();
  shared_ptr<Keyframe> new_keyframe  = make_shared<Keyframe>(Vector3f::Zero(),
                                                             Quaternionf::Identity(),
                                                             left_ptr->image,
                                                             left_features_2d,
                                                             vector<cv::Point3f>(num_features));
    
  // Populate 3d features (used when processing next keyframe)
  triangulate_stereo(new_keyframe->features_3d,
                     new_keyframe->features_2d,
                     new_keyframe->image,
                     right_ptr->image);

  // Use PnP to get transform between last keyframe and this one
  solvePnPRansac(last_keyframe->features_3d,
                 tracked_features,
                 left_pmat.colRange(0, 3),
                 cv::Mat::zeros(4, 1, CV_32F),
                 rvec, tvec, true);

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Matrix3f rmat_eigen;
  Vector3f tvec_eigen;
  cv::cv2eigen(rmat, rmat_eigen);
  cv::cv2eigen(tvec, tvec_eigen);

  // Get absolute pose from relative pose
  new_keyframe->orientation = last_keyframe->orientation*rmat_eigen.transpose();
  new_keyframe->position = last_keyframe->position - last_keyframe->orientation*rmat_eigen.transpose()*tvec_eigen;

  last_keyframe = new_keyframe;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "vo_node");
  ros::NodeHandle n("~");

  // KITTI topic
  /*
  message_filters::Subscriber<sensor_msgs::Image> left_sub(n, "/leftImage", 3);
  message_filters::Subscriber<sensor_msgs::Image> right_sub(n, "/rightImage", 3);
  */

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
  message_filters::Subscriber<sensor_msgs::Image> left_sub(n, "/multisense_sl/camera/left/image_raw", 3);
  message_filters::Subscriber<sensor_msgs::Image> right_sub(n, "/multisense_sl/camera/right/image_raw", 3);

  message_filters::Synchronizer<ApproxSyncPolicy> sync(ApproxSyncPolicy(10), left_sub, right_sub);
  sync.registerCallback(boost::bind(&handle_images, _1, _2));

  tf2_ros::TransformBroadcaster br;

  //ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("features", 1);
  ros::Publisher path_pub = n.advertise<nav_msgs::Path>("/vo/path", 1);
  nav_msgs::Path path;
  path.header.frame_id = "world";

  left_pmat = cv::Mat::eye(3, 4, CV_32F);
  left_pmat.at<float>(0, 0) = focal_length;
  left_pmat.at<float>(0, 2) = cx;
  left_pmat.at<float>(1, 1) = focal_length;
  left_pmat.at<float>(1, 2) = cy;
  right_pmat = left_pmat.clone();
  right_pmat.at<float>(0, 3) = focal_length*stereo_sep;

  ros::Rate r(20);
  while (ros::ok()) {
    if (got_first_image) {
      geometry_msgs::TransformStamped pose;
      pose.transform.rotation.w = last_keyframe->orientation.w();
      pose.transform.rotation.x = last_keyframe->orientation.x();
      pose.transform.rotation.y = last_keyframe->orientation.y();
      pose.transform.rotation.z = last_keyframe->orientation.z();

      pose.transform.translation.x = last_keyframe->position(0);
      pose.transform.translation.y = last_keyframe->position(1);
      pose.transform.translation.z = last_keyframe->position(2);

      pose.header.stamp = ros::Time::now();
      pose.header.frame_id = "world";
      pose.child_frame_id = "vo_pose";

      br.sendTransform(pose);

      geometry_msgs::PoseStamped gpose;
      gpose.pose.orientation.w = last_keyframe->orientation.w();
      gpose.pose.orientation.x = last_keyframe->orientation.x();
      gpose.pose.orientation.y = last_keyframe->orientation.y();
      gpose.pose.orientation.z = last_keyframe->orientation.z();

      gpose.pose.position.x = last_keyframe->position(0);
      gpose.pose.position.y = last_keyframe->position(1);
      gpose.pose.position.z = last_keyframe->position(2);

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

      size_t num_features = last_keyframe->features_3d.size();
      for (size_t i = 0; i < num_features; ++i) {
        geometry_msgs::Point p;
        p.x = keyframe->features_3d[i].x;
        p.y = keyframe->features_3d[i].y;
        p.z = keyframe->features_3d[i].z;

        marker.points.push_back(p);
      }
 
      marker_pub.publish(marker);
      */
    }

    r.sleep();
    ros::spinOnce();
  }

  return 0;
}
