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
##     docker run  -it  --gpus all  --entrypoint /bin/bash  -v /tmp/.X11-unix:/tmp/.X11-unix  -v /opt/shared/uol/developer/github.lcas/drydock_ros/scene_analyser/:/opt/py3_ws/src/drydock_ros/drydock_ros/scene_analyser  -v /opt/shared/uol/developer/model/:/root/scene_analyser/model  --net=host  -e DISPLAY=$DISPLAY  -v ${HOME}/.Xauthority:/root/.Xauthority  ros_scene_analyser

#FROM  melodic_detectron2:local
FROM  lcas.lincoln.ac.uk/lcas/fastpick_semantic_segmentation_server:melodic



## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## update apt-get repositories, upgrade
## existing software & install
## additional software / libs / tools
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~

RUN  apt-get update  &&  apt-get -y upgrade  &&  apt-get install -y  nano  curl  net-tools  lsb-release  git  wget  less  build-essential  libtool  &&  rm -rf /var/lib/apt/lists/*



## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## X11 support
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
RUN  apt-get install -y x11-common
RUN  useradd -ms /bin/bash user
# default value for display. works for most environments, but not all
ENV  DISPLAY :0
#USER  user



## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## .bashrc modifications
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
RUN  cat /opt/py3_ws/src/drydock_ros/scene_analyser/docker_extra/bashrc.txt >> /root/.bashrc



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
COPY  scene_analyser/docker_extra/cv_bridge.build.sh /opt/cv_bridge_ws/cv_bridge.build.sh
#RUN  catkin config -DPYTHON_EXECUTABLE=/opt/conda/bin/python3 -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.6m -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.6m.so  -DSETUPTOOLS_DEB_LAYOUT=OFF
RUN  chmod +x ./cv_bridge.build.sh  &&  ./cv_bridge.build.sh

# 4) add to .bashrc to enable our new cv_bridge module in future calls
RUN  echo "source /opt/cv_bridge_ws/install/setup.bash  --extend" >> ~/.bashrc



## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## building octomap-python
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
#WORKDIR  /root
#RUN  git clone --recursive https://github.com/wkentaro/octomap-python.git
#WORKDIR  /root/octomap-python
#RUN  python ./setup.py build  &&  python setup.py install


## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## adding the pytorch model
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
#RUN  mkdir -p /root/scene_analyser/model  &&  wget https://files.lar.lincoln.ac.uk/s/iKGCwFz8PaNn4GX/download/fp_ss_model.pth /root/scene_analyser/model/
RUN  mkdir -p /root/scene_analyser/model  &&  wget  -q -O /root/scene_analyser/model/fp_model.pth  https://lncn.ac/fp_model.pth



## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## last minute hacks
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
RUN  chmod +x /opt/py3_ws/src/drydock_ros/scene_analyser/scripts/*.py
RUN  pip install octomap-python
#RUN  apt-get update  &&  apt-get install -y libdynamicedt3d1.8 ros-melodic-octomap ros-melodic-dynamic-edt-3d ros-melodic-octomap-ros ros-melodic-octovis ros-melodic-octomap-msgs  &&  rm -rf /var/lib/apt/lists/*
RUN  apt-get update  &&  apt-get install -y libdynamicedt3d1.8 ros-melodic-octomap-msgs  &&  rm -rf /var/lib/apt/lists/*


## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## CMD
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~


CMD /bin/bash
#CMD ["/bin/bash", "-c", "tmux"]
#CMD ["/bin/bash", "-c", "(/usr/local/lib/remote_manager/vnc_runner/noVNC/utils/launch.sh --vnc localhost:5901 --listen 6080 &); rm -f /tmp/.X1-lock /tmp/.X11-unix/X1; (vncserver -localhost yes -depth 24 -geometry 1280x720 -SecurityTypes None -fg :1 &);    ((source /detectron2_venv/bin/activate; source /opt/ros/melodic/setup.bash; source /opt/py3_ws/install/setup.bash; ROS_PYTHON_VERSION=3 roslaunch scene_analyser scene_analyser.launch)&) ; sleep 5; (conda deactivate; export PATH=/usr/bin:$PATH; source /opt/ros/melodic/setup.bash; source /opt/py3_ws/install/setup.bash; source /opt/py2_ws/install/setup.bash; ROS_PYTHON_VERSION=3 roslaunch /opt/py3_ws/install/share/drydock_ros/launch/host.launch)"]
