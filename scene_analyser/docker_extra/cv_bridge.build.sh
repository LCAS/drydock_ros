#!/bin/bash

source ~/.bashrc

python --version
echo ""
echo "======-======-======-======"
/opt/ros/melodic/env.sh  echo "ROS version: `rosversion -d`"
/opt/ros/melodic/env.sh  echo "ROS_MASTER_URI=${ROS_MASTER_URI}"
/opt/ros/melodic/env.sh  echo "ROS_HOSTNAME=${ROS_HOSTNAME}"
/opt/ros/melodic/env.sh  echo "ROS_IP=${ROS_IP}"
/opt/ros/melodic/env.sh  echo "ROS_PACKAGE_PATH:${ROS_PACKAGE_PATH}"
echo "======-======-======-======"


/opt/ros/melodic/env.sh catkin config \
	-DPYTHON_EXECUTABLE=/opt/conda/bin/python3 \
	-DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.6m \
	-DPYTHON_LIBRARY=/opt/conda/lib/libpython3.6m.so \
	-DSETUPTOOLS_DEB_LAYOUT=OFF

/opt/ros/melodic/env.sh  catkin config --install

/opt/ros/melodic/env.sh  catkin build
