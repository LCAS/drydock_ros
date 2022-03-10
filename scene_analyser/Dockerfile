## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## ROS SCENE ANALYSER 
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
##
## this dockerfile is for the scene analyser ros framework. it provides a ros node & action server to use the scene analyser
## (aka MaskPredictor) in a ros environment
##
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## created and maintained by:
##   johann dichtl (jdichtl@lincoln.ac.uk)
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
##
## to build the docker image use (from parent directory):
##   docker build  -f scene_analyser/Dockerfile  -t ros_scene_analyser  .
##
## to run the docker image use (adjust paths as necessary):
##   basic command:
##     docker run  -it  --gpus all  --entrypoint /bin/bash  --net=host  -e DISPLAY  -v ${HOME}/.Xauthority:/root/.Xauthority  ros_scene_analyser
##
##   optional mount volumes, e.g.:
##     docker run  -it  --gpus all  --entrypoint /bin/bash  --net=host  -e DISPLAY  -v ${HOME}/.Xauthority:/root/.Xauthority  -v /path/to/local/repo/MaskPredictor/:/root/scene_analyser/  ros_scene_analyser
##     docker run  -it  --gpus all  --entrypoint /bin/bash  -v /opt/shared/uol/developer/github.lcas/drydock_ros/scene_analyser/:/opt/fastpick/src/scene_analyser  -v /opt/shared/uol/developer/model/:/root/scene_analyser/model  --net=host  -e DISPLAY  -v ${HOME}/.Xauthority:/root/.Xauthority  ros_scene_analyser
##
## if you encounter the error
## > Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
## then you are missing the nvidia driver for docker. the installation instructions for that can be found here:
##   https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html


FROM  melodic_detectron2:local



## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## update apt-get repositories, upgrade
## existing software & install
## additional software / libs / tools
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
RUN  apt-get update  &&  apt-get -y upgrade
RUN  apt-get install -y  nano  curl  python3-pip  net-tools  lsb-release  git  wget  less  build-essential  libtool  tmux  openssh-client


## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## X11 support
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
RUN  apt-get install -y x11-common
RUN  useradd -ms /bin/bash user
ENV  DISPLAY :0
#USER  user


## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## install cv_bridge for python3
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
#RUN  apt-get install -y python3-numpy
RUN  mkdir -p /opt/cvbridge_build_ws/src
WORKDIR  /opt/cvbridge_build_ws/src
RUN  git clone -b noetic https://github.com/ros-perception/vision_opencv.git
RUN  sed -i 's/python37/python3/g' vision_opencv/cv_bridge/CMakeLists.txt
WORKDIR  /opt/cvbridge_build_ws
RUN  sed -i 's/--install-layout=deb/ /g'  /opt/ros/melodic/share/catkin/cmake/python.cmake
RUN  echo "++++++++ python version ++++++++"
RUN  python --version
RUN  /opt/ros/melodic/env.sh  catkin config -DPYTHON_EXECUTABLE=/opt/conda/bin/python3 \
         -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.6m \
         -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.6m.so
RUN  /opt/ros/melodic/env.sh  catkin config --install
RUN  /opt/ros/melodic/env.sh  catkin build cv_bridge


## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## ROS scene_analyser code
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## Note: we have a ros workspace at /opt/py3_ws from our parent image
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
WORKDIR  /opt/py3_ws
COPY  scene_analyser  /opt/py3_ws/src/scene_analyser
# modify .bashrc to work with our workspace and also adds ros info to terminal
RUN  cat src/scene_analyser/docker_extra/bashrc.txt >> /root/.bashrc
#RUN  /opt/ros/melodic/env.sh  rosdep install -y -i --from-paths .
#RUN  /opt/ros/melodic/env.sh  catkin_make
#RUN  echo "source /opt/cvbridge_build_ws/install/setup.bash --extend" >> /root/.bashrc


## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## add the scene analyser code
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
WORKDIR  /opt/py3_ws/src/scene_analyser
#RUN  apt-get install -y ros-melodic-cv-bridge
#RUN  pip3 install rospkg opencv-python

WORKDIR  /opt/py3_ws
RUN  /opt/ros/melodic/env.sh  catkin_make

## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
## switch to final directory and start a terminal
## ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
WORKDIR  /opt/py3_ws
RUN  /opt/py3_ws/devel/env.sh pip3 install pyyaml rospkg



CMD /bin/bash
#CMD ["/bin/bash", "-c", "((source /detectron2_venv/bin/activate; source /opt/ros/melodic/setup.bash; source /opt/py3_ws/install/setup.bash; ROS_PYTHON_VERSION=3 roslaunch scene_analyser scene_analyser.launch)&) ; sleep 5; (conda deactivate; export PATH=/usr/bin:$PATH; source /opt/ros/melodic/setup.bash; source /opt/py3_ws/install/setup.bash; source /opt/py2_ws/install/setup.bash; ROS_PYTHON_VERSION=3 roslaunch /opt/py3_ws/install/share/drydock_ros/launch/host.launch)"]


