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
#include "feature_tracker.hpp"
#include <string>

using namespace Eigen;
using namespace std;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
                                                        sensor_msgs::Image> ApproxSyncPolicy;

typedef function<void (const sensor_msgs::ImageConstPtr& left_msg, const sensor_msgs::ImageConstPtr& right_msg)> StereoImageHandler;

static const float parallax_thresh = 20;
static const float min_feature_distance = 30;
static const size_t image_queue_size = 5;
static const size_t sliding_window_size = 5;

/*
 * triangulate_stereo: triangulates the 3d positions of 2d features
 * ARGUMENTS
 * features_3d: populated with the 3d positions of valid_features_2d
 * valid_features_2d: populated with elements of left_features_2d with nonzero disparity
 * the left and right images, which is a condition necessary for
 * triangulation. This vector is populated with all
 * 2d feature positions that were able to take part in triangulation
 * left_features_2d: 2d positions in left image to triangulate
 * left: left image
 * right: right image
 * camera_pose: 4x4 matrix transforming points in camera frame to
 *              world frame
 * focal: focal length
 * cx: x pixel offset
 * cy: y pixel offset
 * baseline: horizontal separation between cameras in stereo pair
 * REQUIRES: features_3d.size() == 0 
 */
void triangulate_stereo(vector<cv::Point3f>& features_3d,
                        vector<cv::Point2f>& valid_features_2d,
                        const vector<cv::Point2f>& left_features_2d,
                        const cv::Mat& left,
                        const cv::Mat& right,
                        const cv::Mat& camera_pose,
                        const float focal,
                        const float cx,
                        const float cy,
                        const float baseline) {

  // Get disparity 
  cv::Mat disparity;
  cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(16*3, 21);
  sbm->compute(left, right, disparity);
  disparity.convertTo(disparity, CV_32F, 1.0/16);

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

class handle_images {
  private:
    shared_ptr<BundleAdjuster> bundle_adjuster;
    shared_ptr<FeatureTracker> feature_tracker;
    cv::Mat camera_matrix;
    float baseline;

    // We use the previous transform (i.e. the previous solvePnP output) as the
    // initial guess for the next PnP solution
    cv::Mat rvec;
    cv::Mat tvec;

    double last_time;

  public:
    /*
     * handle_images: constructor
     * ARGUMENTS
     * cam_mat: camera_matrix;
     * adjuster: BundleAdjuster object to add keyframes to 
     * bline: stereo baseline
     */
    handle_images(cv::Mat cam_mat, 
                  shared_ptr<FeatureTracker> tracker,
                  shared_ptr<BundleAdjuster> adjuster,
                  float bline) : camera_matrix(cam_mat),
                                 feature_tracker(tracker),
                                 bundle_adjuster(adjuster),
                                 baseline(bline) {}

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

      double time = left_msg->header.stamp.toSec();
      if (time - last_time < 0.05) {
        return;
      }
      last_time = time;

      // Detect features in left image
      vector<cv::Point2f> detected_features;

      cv::goodFeaturesToTrack(left_ptr->image, detected_features, 300, 0.1, min_feature_distance);
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
                           left_ptr->image,
                           right_ptr->image,
                           cv::Mat::eye(4, 4, CV_32F),
                           camera_matrix.at<float>(0, 0),
                           camera_matrix.at<float>(0, 2),
                           camera_matrix.at<float>(1, 2),
                           baseline);

        shared_ptr<Keyframe> new_keyframe = make_shared<Keyframe>(Vector3f::Zero(),
                                                                  Quaternionf::Identity(),
                                                                  left_ptr->image,
                                                                  valid_features_2d,
                                                                  features_3d,
                                                                  vector<size_t>());

        bundle_adjuster->add_keyframe(new_keyframe);

        feature_tracker->init(left_ptr->image, valid_features_2d,
                              new_keyframe->feature_ids);

        tvec = cv::Mat::zeros(3, 1, CV_32F);
        rvec = cv::Mat::zeros(3, 1, CV_32F);

        return;
      }

      float av_parallax = 0;
      float percent_lost = 0;
      feature_tracker->track_features(av_parallax, percent_lost, left_ptr->image, true);
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
                                                                left_ptr->image,
                                                                vector<cv::Point2f>(num_inliers),
                                                                vector<cv::Point3f>(num_inliers),
                                                                vector<size_t>(num_inliers));

      // Populate new keyframe's features with inliers
      for (size_t i = 0; i < num_inliers; i++) {
        size_t idx = inlier_indices.at<int>(i, 0);
        new_keyframe->feature_ids[i] = tracked_ids[idx];
        new_keyframe->features_2d[i] = tracked_features[idx];
        new_keyframe->features_3d[i] = tracked_world_points[idx];
      }

      // Now we need to triangulate newly detected features. However, many of these
      // features are probably the same ones we're already tracking. Therefore, filter
      // those out, then triangulate
      size_t num_features = detected_features.size();
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
      vector<cv::Point2f> valid_new_features;

      cv::Mat hmat = cv::Mat::eye(4, 4, CV_32F);
      cv::Mat tmp = rmat.t();
      tmp.copyTo(hmat(cv::Rect(0, 0, 3, 3)));
      tmp = -tmp*tvec;
      tmp.copyTo(hmat(cv::Rect(3, 0, 1, 3)));

      // Triangulate 3d points for detected features
      triangulate_stereo(new_features_3d,
                         valid_new_features,
                         new_features,
                         left_ptr->image,
                         right_ptr->image,
                         hmat,
                         camera_matrix.at<float>(0, 0),
                         camera_matrix.at<float>(0, 2),
                         camera_matrix.at<float>(1, 2),
                         baseline);

      new_keyframe->features_2d.insert(new_keyframe->features_2d.end(), 
                                       make_move_iterator(valid_new_features.begin()),
                                       make_move_iterator(valid_new_features.end()));

      new_keyframe->features_3d.insert(new_keyframe->features_3d.end(), 
                                       make_move_iterator(new_features_3d.begin()),
                                       make_move_iterator(new_features_3d.end()));

      bundle_adjuster->add_keyframe(new_keyframe);

      feature_tracker->draw_track();
      feature_tracker->init(left_ptr->image, new_keyframe->features_2d, new_keyframe->feature_ids);
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
  cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_32F);
  camera_matrix.at<float>(0, 0) = focal_length;
  camera_matrix.at<float>(0, 2) = cx;
  camera_matrix.at<float>(1, 1) = focal_length;
  camera_matrix.at<float>(1, 2) = cy;

  CameraInfo info = {focal_length, cx, cy, 0, 0, 0, 0, baseline};

  shared_ptr<BundleAdjuster> bundle_adjuster = make_shared<BundleAdjuster>(sliding_window_size, info);
  shared_ptr<FeatureTracker> feature_tracker = make_shared<FeatureTracker>();

  // Synchronize subscribers to left and right images
  message_filters::Synchronizer<ApproxSyncPolicy> sync(ApproxSyncPolicy(10), left_sub, right_sub);
  StereoImageHandler image_handler = handle_images(camera_matrix, feature_tracker, bundle_adjuster, baseline);
  sync.registerCallback(bind(image_handler, _1, _2));

  tf2_ros::TransformBroadcaster br;

  ros::Publisher path_pub = n.advertise<nav_msgs::Path>("/vo/path", 1);
  nav_msgs::Path path;
  path.header.frame_id = "world";

  //ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("/vo/features", 1);

  ros::Publisher tracking_pub = n.advertise<sensor_msgs::Image>("/feature_tracking", 1);
  cv_bridge::CvImagePtr tracking_ptr = boost::make_shared<cv_bridge::CvImage>();
  tracking_ptr->encoding = "rgb8";

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

      // Publish feature tracks
      tracking_ptr->image = feature_tracker->get_drawing();
      tracking_pub.publish(tracking_ptr->toImageMsg());

      /*
      // Display features in rviz
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
