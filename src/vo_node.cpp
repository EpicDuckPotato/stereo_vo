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
#include "image_processor.hpp"
#include <string>
#include <queue>

using namespace Eigen;
using namespace std;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
                                                        sensor_msgs::Image> ApproxSyncPolicy;

typedef function<void (const sensor_msgs::ImageConstPtr& left_msg, const sensor_msgs::ImageConstPtr& right_msg)> StereoImageHandler;

static const float parallax_thresh = 20;
static const float min_feature_distance = 30;
static const size_t image_queue_size = 5;
static const size_t sliding_window_size = 5;

class handle_images {
  private:
    shared_ptr<queue<StereoPair>> image_queue;
    double drop_time;

  public:
    /*
     * handle_images: constructor
     * ARGUMENTS
     * image_queue: pointer to queue onto which we should push stereo pairs
     * drop_time: what is the minimum interval between images that we care about?
     */
    handle_images(shared_ptr<queue<StereoPair>> image_queue,
                  double drop_time) : image_queue(image_queue),
                                      drop_time(drop_time) {}

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
      double time = left_msg->header.stamp.toSec();
      if (!image_queue->empty() && time - image_queue->back().t < drop_time) {
        return;
      }

      cv_bridge::CvImagePtr left_ptr = cv_bridge::toCvCopy(*left_msg, sensor_msgs::image_encodings::MONO8);
      cv_bridge::CvImagePtr right_ptr = cv_bridge::toCvCopy(*right_msg, sensor_msgs::image_encodings::MONO8);

      image_queue->push(StereoPair(left_ptr->image, right_ptr->image, time));
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
  ImageProcessor image_processor(camera_matrix,
                                 feature_tracker,
                                 bundle_adjuster,
                                 baseline,
                                 min_feature_distance,
                                 parallax_thresh);

  // Synchronize subscribers to left and right images
  message_filters::Synchronizer<ApproxSyncPolicy> sync(ApproxSyncPolicy(10), left_sub, right_sub);
  shared_ptr<queue<StereoPair>> image_queue = make_shared<queue<StereoPair>>();
  StereoImageHandler image_handler = handle_images(image_queue, 0.05);
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
    while (!image_queue->empty()) {
      image_processor.process(image_queue->front());
      image_queue->pop();
    }

    if (bundle_adjuster->get_last_keyframe() != nullptr) {
      bundle_adjuster->bundle_adjust();
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
    }

    r.sleep();
    ros::spinOnce();
  }

  return 0;
}
