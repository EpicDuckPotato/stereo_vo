#include <iostream>
#include <fstream>
#include <sensor_msgs/Image.h>
#include "ros/ros.h"
#include <Eigen/Dense>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Path.h>

using namespace std;
using namespace Eigen;

static const string data_path = "/home/noopygbhat/CMU/biorobotics/kitti_odom_datasets/";

int main(int argc, char** argv) {
  int sequence_int;
  ros::init(argc, argv, "kitti_node");
  ros::NodeHandle n("~");

  n.getParam("sequence", sequence_int);
  string sequence = to_string(sequence_int);
  if (sequence_int < 10) {
    sequence = "0" + sequence;
  }

  ros::Publisher left_pub = n.advertise<sensor_msgs::Image>("/leftImage", 100);
  ros::Publisher right_pub = n.advertise<sensor_msgs::Image>("/rightImage", 100);

  tf2_ros::TransformBroadcaster br;

  cv::Mat left, right;

  // Ground truth file
  ifstream gt(data_path + "data_odometry_poses/dataset/poses/" + sequence + ".txt");

  Matrix3d R;
  Vector3d t;

  ros::Publisher path_pub = n.advertise<nav_msgs::Path>("/ground_truth/path", 1);
  nav_msgs::Path path;
  path.header.frame_id = "world";

  ros::Rate r(11);
  for (size_t i = 0;
       gt >> R(0, 0) >> R(0, 1) >> R(0, 2) >> t(0) >> 
             R(1, 0) >> R(1, 1) >> R(1, 2) >> t(1) >> 
             R(2, 0) >> R(2, 1) >> R(2, 2) >> t(2);
       i++)
  {
    if(ros::ok())
    {
      // Publish images
      string prefix;
      if (i < 10) {
        prefix = "00000";
      } else if (i < 100) {
        prefix = "0000";
      } else if (i < 1000) {
        prefix = "000";
      } else {
        prefix = "00";
      }

      left = cv::imread(data_path + sequence + "/image_0/" + prefix + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
      right = cv::imread(data_path + sequence + "/image_1/" + prefix + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
      sensor_msgs::ImagePtr left_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", left).toImageMsg();
      left_msg->header.stamp = ros::Time::now();

      sensor_msgs::ImagePtr right_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", right).toImageMsg();
      right_msg->header.stamp = left_msg->header.stamp;

      left_pub.publish(left_msg);
      right_pub.publish(right_msg);

      // Publish ground truth
      Quaterniond q(R);

      geometry_msgs::TransformStamped pose;
      pose.transform.rotation.w = q.w();
      pose.transform.rotation.x = q.x();
      pose.transform.rotation.y = q.y();
      pose.transform.rotation.z = q.z();

      pose.transform.translation.x = t(0);
      pose.transform.translation.y = t(1);
      pose.transform.translation.z = t(2);

      pose.header.stamp = right_msg->header.stamp;
      pose.header.frame_id = "world";
      pose.child_frame_id = "ground_truth";

      br.sendTransform(pose);

      geometry_msgs::PoseStamped gpose;
      gpose.pose.orientation.w = q.w();
      gpose.pose.orientation.x = q.x();
      gpose.pose.orientation.y = q.y();
      gpose.pose.orientation.z = q.z();

      gpose.pose.position.x = t(0);
      gpose.pose.position.y = t(1);
      gpose.pose.position.z = t(2);

      gpose.header.stamp = pose.header.stamp;
      gpose.header.frame_id = "ground_truth_pose";

      path.poses.push_back(gpose);
      path.header.stamp = gpose.header.stamp;
      path_pub.publish(path);
      r.sleep();
    } else {
      break;
    }
  }
  return 0;
}
