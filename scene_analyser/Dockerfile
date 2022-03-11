## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## ROS SCENE ANALYSER
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
##
## to build the docker image use (from the git root parent directory):
##   docker build  -f drydock_ros/scene_analyser/Dockerfile  -t ros_scene_analyser  .
##
## to run the docker image use (adjust paths as necessary):
##   basic command:
##     docker run  -it  ros_scene_analyser
##
##     docker run  -it  --gpus all  --entrypoint /bin/bash  -v /opt/shared/uol/developer/github.lcas/drydock_ros/scene_analyser/:/opt/py3_ws/src/drydock_ros/drydock_ros/scene_analyser  -v /opt/shared/uol/developer/model/:/root/scene_analyser/model  --net=host  -e DISPLAY  -v ${HOME}/.Xauthority:/root/.Xauthority  ros_scene_analyser

FROM  melodic_detectron2:local



## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## update apt-get repositories, upgrade
## existing software & install
## additional software / libs / tools
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~

RUN  apt-get update  &&  apt-get -y upgrade  &&  apt-get install -y  nano  curl  net-tools  lsb-release  git  wget  less  build-essential  libtool  &&   rm -rf /var/lib/apt/lists/*



## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## X11 support
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
RUN  apt-get install -y x11-common
RUN  useradd -ms /bin/bash user
ENV  DISPLAY :0
#USER  user



## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## .bashrc modifications
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
WORKDIR  /opt/py3_ws
RUN  cat src/drydock_ros/drydock_ros/scene_analyser/docker_extra/bashrc.txt >> /root/.bashrc



## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## cv_bridge custom build with python3
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~

# 1) clone to custon workspace
RUN  mkdir -p /opt/cv_bridge_ws/src
WORKDIR  /opt/cv_bridge_ws/src
RUN  git clone https://github.com/ros-perception/vision_opencv

# 2) modify to work with python 3.6
RUN  sed -i 's/python37/python3/g' ./vision_opencv/cv_bridge/CMakeLists.txt

# 3) build via catkin_build
WORKDIR  /opt/cv_bridge_ws
COPY  drydock_ros/scene_analyser/docker_extra/cv_bridge.build.sh /opt/cv_bridge_ws/cv_bridge.build.sh
#RUN  catkin config -DPYTHON_EXECUTABLE=/opt/conda/bin/python3 -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.6m -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.6m.so  -DSETUPTOOLS_DEB_LAYOUT=OFF
RUN  chmod +x ./cv_bridge.build.sh  &&  ./cv_bridge.build.sh

# 4) add to .bashrc to enable our new cv_bridge module in future calls
RUN  echo "source /opt/cv_bridge_ws/install/setup.bash  --extend" >> ~/.bashrc



## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## CMD
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~


CMD /bin/bash
