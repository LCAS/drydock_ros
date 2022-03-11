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



CMD /bin/bash
