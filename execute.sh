#!/bin/bash

# First, execute a roscore, since even though roslaunch will run one in the background
# if it doesn't already exist, the problem is that if it crashes, everything dies.
# It is usually better to run a roscore first, then do roslaunches. That way, roslaunches
# use the separate roscore.

#roscore


# You also need to source the package setup to make sure ROS can find your package.

#source devel/setup.bash


# Now you can launch the package's launch file.

#roslaunch epic epic_rviz_maze_1.launch


# On Robot:
export ROS_MASTER_URI=http://192.168.0.229:11311
export ROS_IP=192.168.0.229

# On Host:
export ROS_MASTER_URI=http://192.168.0.229:11311
export ROS_IP=192.168.0.226

