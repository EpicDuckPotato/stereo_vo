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
#include <string>

using namespace Eigen;
using namespace std;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
                                                        sensor_msgs::Image> ApproxSyncPolicy;

typedef function<void (const sensor_msgs::ImageConstPtr& left_msg, const sensor_msgs::ImageConstPtr& right_msg)> StereoImageHandler;

static const float parallax_thresh = 20;
static const float min_feature_distance = 30;
static const size_t image_queue_size = 3;
static const size_t sliding_window_size = 5;

/*
 * triangulate_stereo: triangulates the 3d positions of 2d features
 * ARGUMENTS
 * features_3d: populated with the 3d positions of any features in
 * left_keypoints that can be triangulated
 * matched_features_2d: not all features will be present in both
 * the left and right images, which is a condition necessary for
 * triangulation. This vector is populated with all
 * 2d feature positions that were able to take part in triangulation
 * left_keypoints: keypoints in left image to triangulate
 * right_keypoints: keypoints in right image to triangulate
 * left_pmat: projection matrix for left camera in stereo pair
 * focal_length: focal length
 * baseline: baseline
 * REQUIRES: features_3d.size() == 0 && matched_features_2d.size() == 0
 * ENSURES: features_3d.size() == matched_features_2d.size()
 */
void triangulate_stereo(vector<cv::Point3f>& features_3d,
                        vector<cv::Point2f>& matched_features_2d,
                        const vector<cv::KeyPoint>& left_keypoints,
                        const vector<cv::KeyPoint>& right_keypoints,
                        const cv::Mat& left_descriptors,
                        const cv::Mat& right_descriptors,
                        const cv::Mat& left_pmat,
                        float focal_length,
                        float baseline) {

  cv::Mat right_pmat = left_pmat.clone();
  right_pmat.at<float>(0, 3) -= focal_length*baseline;

  cv::BFMatcher matcher(cv::NORM_HAMMING);
  vector<cv::DMatch> matches;
  matcher.match(left_descriptors, right_descriptors, matches);
  vector<cv::Point2f> right_features_2d;
  for (cv::DMatch match : matches) {
    int left_index = match.queryIdx;
    int right_index = match.trainIdx;
    if (abs(left_keypoints[left_index].pt.y - right_keypoints[right_index].pt.y) < 2.0 &&
        match.distance > 2.0) {
      matched_features_2d.push_back(left_keypoints[left_index].pt);
      right_features_2d.push_back(right_keypoints[right_index].pt);
    }
  }

  // Triangulate 3d positions of features that could be tracked
  size_t num_features = matched_features_2d.size();
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

class handle_images {
  private:
    shared_ptr<BundleAdjuster> bundle_adjuster;
    cv::Mat camera_matrix;
    float baseline;

    // We use the previous transform (i.e. the previous solvePnP output) as the
    // initial guess for the next PnP solution
    cv::Mat rvec;
    cv::Mat tvec;

    cv::Ptr<cv::ORB> detector;
    cv::Mat orb_mask;

  public:
    /*
     * handle_images: constructor
     * ARGUMENTS
     * cam_mat: camera_matrix;
     * adjuster: BundleAdjuster object to add keyframes to 
     * bline: stereo baseline
     */
    handle_images(cv::Mat cam_mat, 
                  shared_ptr<BundleAdjuster> adjuster,
                  float bline) : camera_matrix(cam_mat),
                                 bundle_adjuster(adjuster),
                                 baseline(bline) {
      detector = cv::ORB::create(400, 1.2f, 3, 21, 0, 2, cv::ORB::HARRIS_SCORE, 31, 40); 
    }

    /*
     * operator (): as we get image pairs from ROS, we use them for processing only if features
     * from the last keyframe have sufficient parallax with this pair's left image.
     * If so, use previous keyframe's 3d feature positions to solve for current left
     * camera pose via PnP
     * ARGUMENTS
     * left_msg: left image message
     * right_msg: right image message
     */
    void operator () (const sensor_msgs::ImageConstPtr& left_msg,
                      const sensor_msgs::ImageConstPtr& right_msg) { 
      cv_bridge::CvImagePtr left_ptr = cv_bridge::toCvCopy(*left_msg, sensor_msgs::image_encodings::MONO8);
      cv_bridge::CvImagePtr right_ptr = cv_bridge::toCvCopy(*right_msg, sensor_msgs::image_encodings::MONO8);

      if (orb_mask.empty()) {
        orb_mask = 255*cv::Mat::ones(left_ptr->image.rows, left_ptr->image.cols, CV_8UC1);
      }

      // Detect features in left image
      vector<cv::Point2f> detected_features;

      vector<cv::KeyPoint> left_keypoints;
      vector<cv::KeyPoint> right_keypoints;
      cv::Mat left_descriptors;
      cv::Mat right_descriptors;

      detector->detectAndCompute(left_ptr->image, orb_mask, left_keypoints, left_descriptors); 
      detector->detectAndCompute(right_ptr->image, orb_mask, right_keypoints, right_descriptors); 

      shared_ptr<Keyframe> last_keyframe = bundle_adjuster->get_last_keyframe();

      // If we don't have any keyframes, make this the first one
      if (last_keyframe == nullptr) {
        vector<cv::Point3f> features_3d;
        vector<cv::Point2f> matched_features_2d;

        triangulate_stereo(features_3d,
                           matched_features_2d,
                           left_keypoints,
                           right_keypoints,
                           left_descriptors,
                           right_descriptors,
                           camera_matrix,
                           camera_matrix.at<float>(0, 0),
                           baseline);

        shared_ptr<Keyframe> new_keyframe = make_shared<Keyframe>(Vector3f::Zero(),
                                                                  Quaternionf::Identity(),
                                                                  left_ptr->image,
                                                                  matched_features_2d,
                                                                  features_3d,
                                                                  vector<size_t>());

        bundle_adjuster->add_keyframe(new_keyframe);
        
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
      // check average parallax. If it's less than a threshold, don't use this frame as a keyframe
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

      // Use PnP to get pose of world frame with respect to the current camera frame
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
      num_features = left_keypoints.size();
      vector<cv::KeyPoint> new_left_keypoints;
      cv::Mat new_left_descriptors;
      for (size_t i = 0; i < num_features; i++) {
        bool tracked = false;
        for (size_t j = 0; j < num_inliers; j++) {
          float dx = left_keypoints[i].pt.x - new_keyframe->features_2d[j].x;
          float dy = left_keypoints[i].pt.y - new_keyframe->features_2d[j].y;
          if (sqrt(dx*dx + dy*dy) < min_feature_distance) {
            tracked = true;
            break;
          }
        }
        if (!tracked) {
          new_left_keypoints.push_back(left_keypoints[i]);
          new_left_descriptors.push_back(left_descriptors.row(i));
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
                         new_left_keypoints,
                         right_keypoints,
                         new_left_descriptors,
                         right_descriptors,
                         camera_matrix*hmat,
                         camera_matrix.at<float>(0, 0),
                         baseline);

      new_keyframe->features_2d.insert(new_keyframe->features_2d.end(), 
                                       make_move_iterator(matched_new_features.begin()),
                                       make_move_iterator(matched_new_features.end()));

      new_keyframe->features_3d.insert(new_keyframe->features_3d.end(), 
                                       make_move_iterator(new_features_3d.begin()),
                                       make_move_iterator(new_features_3d.end()));

      bundle_adjuster->add_keyframe(new_keyframe);
    }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "vo_node");
  ros::NodeHandle n("~");

  // Camera intrinsics
  float focal_length;
  float cx;
  float cy;
  float baseline;

  // ROS info
  string left_topic;
  string right_topic;
  int frame_rate;

  n.getParam("/focal_length", focal_length);
  n.getParam("/cx", cx);
  n.getParam("/cy", cy);
  n.getParam("/baseline", baseline);
  n.getParam("/left_topic", left_topic);
  n.getParam("/right_topic", right_topic);
  n.getParam("/frame_rate", frame_rate);

  message_filters::Subscriber<sensor_msgs::Image> left_sub(n, left_topic, image_queue_size);
  message_filters::Subscriber<sensor_msgs::Image> right_sub(n, right_topic, image_queue_size);

  // Populate camera matrix with intrinsics
  cv::Mat camera_matrix = cv::Mat::eye(3, 4, CV_32F);
  camera_matrix.at<float>(0, 0) = focal_length;
  camera_matrix.at<float>(0, 2) = cx;
  camera_matrix.at<float>(1, 1) = focal_length;
  camera_matrix.at<float>(1, 2) = cy;

  CameraInfo info = {focal_length, cx, cy, 0, 0, 0, 0, baseline};

  shared_ptr<BundleAdjuster> bundle_adjuster = make_shared<BundleAdjuster>(sliding_window_size, info);

  // Synchronize subscribers to left and right images
  message_filters::Synchronizer<ApproxSyncPolicy> sync(ApproxSyncPolicy(10), left_sub, right_sub);
  StereoImageHandler image_handler = handle_images(camera_matrix, bundle_adjuster, baseline);
  sync.registerCallback(bind(image_handler, _1, _2));

  tf2_ros::TransformBroadcaster br;

  ros::Publisher path_pub = n.advertise<nav_msgs::Path>("/vo/path", 1);
  nav_msgs::Path path;
  path.header.frame_id = "world";

  ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("features", 1);

  ros::Rate r(20);
  while (ros::ok()) {
    if (bundle_adjuster->get_last_keyframe() != nullptr) {
      shared_ptr<Keyframe> keyframe = bundle_adjuster->get_last_keyframe();
      Quaternionf orientation = keyframe->orientation.conjugate();
      Vector3f position = orientation*(-keyframe->position);

      // Publish camera position to tf
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

      // Publish path
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

      // Display features in rviz
      /*
      visualization_msgs::Marker marker;
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
      marker.header.stamp = pose.header.stamp;

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

      bundle_adjuster->bundle_adjust();
    }

    r.sleep();
    ros::spinOnce();
  }

  return 0;
}
