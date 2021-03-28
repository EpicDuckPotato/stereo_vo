# stereo_vo
Stereo visual odometry implementation in ROS

Test against the KITTI Vision Benchmark dataset by typing the following in terminal (replacing the kitti number with whichever sequence you want to run). Note, you'll need to change line 16 of kitti_node.cpp to point to wherever you've downloaded the dataset.

roslaunch stereo_vo vo_rviz.launch config:=kitti00

rosrun stereo_vo kitti_node _sequence:=00
