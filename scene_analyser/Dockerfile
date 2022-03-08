# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
# ROS SCENE ANALYSER 
# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
#
# this dockerfile is for the scene analyser ros framework. it provides a ros workspace and python code to use the scene analyser code
# in a ros environment
#
# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
# created and maintained by:
#   johann dichtl (jdichtl@lincoln.ac.uk)
# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
#
# to build the docker image use (from parent directory):
#   docker build  -f scene_analyser/Dockerfile  -t ros_scene_analyser  --build-arg ssh_prv_key="$(cat ~/.ssh/id_ed25519)"  --build-arg ssh_pub_key="$(cat ~/.ssh/id_ed25519.pub)"  .
#
# to run the docker image use (adjust paths as necessary):
#   basic command:
#   docker run  -it  --gpus all  --entrypoint  --net=host  -e DISPLAY  -v ${HOME}/.Xauthority:/root/.Xauthority  ros_scene_analyser
#   optional mount volumes, e.g.:
#   docker run  -it  --gpus all  --entrypoint  --net=host  -e DISPLAY  -v ${HOME}/.Xauthority:/root/.Xauthority  -v /path/to/local/repo/MaskPredictor/:/root/scene_analyser/  ros_scene_analyser
#   docker run  -it  --gpus all  --entrypoint /bin/bash  -v /opt/shared/uol/developer/github.lcas/drydock_ros/scene_analyser/:/opt/fastpick/src/scene_analyser  -v /opt/shared/uol/developer/model/:/root/scene_analyser/model  --net=host  -e DISPLAY  -v ${HOME}/.Xauthority:/root/.Xauthority  ros_scene_analyser
#

FROM  melodic_detectron2:local



# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
# update apt-get repositories, upgrade
# existing software & install
# additional software / libs / tools
# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
RUN  apt-get update  &&  apt-get -y upgrade
RUN  apt-get install -y  nano  curl  python3-pip  net-tools  lsb-release  git  wget  less  build-essential  libtool  tmux  openssh-client
# because python (2.7) is not installed we set an alias here for convenience
#RUN  ln -s /usr/bin/python3 /usr/bin/python


# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
# X11 support
# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
RUN  apt-get install -y x11-common
RUN  useradd -ms /bin/bash user
ENV  DISPLAY :0
#USER  user


# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
# setup ssh public/private key so that we can clone github repositories
# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
ARG ssh_prv_key
ARG ssh_pub_key

# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

# Add the keys and set permissions
RUN echo "$ssh_prv_key" > /root/.ssh/id_ed25519 && \
    echo "$ssh_pub_key" > /root/.ssh/id_ed25519.pub && \
    chmod 600 /root/.ssh/id_ed25519 && \
    chmod 600 /root/.ssh/id_ed25519.pub

RUN  eval "$(ssh-agent)"  &&  ssh-add /root/.ssh/id_ed25519


 

# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
# the ROS scene_analyser code
# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
COPY  scene_analyser  /opt/fastpick/src/scene_analyser
WORKDIR  /opt/fastpick
RUN  cat src/scene_analyser/docker_extra/bashrc.txt >> /root/.bashrc
#RUN  /opt/ros/melodic/env.sh  rosdep install -y -i --from-paths .
RUN  /opt/ros/melodic/env.sh  catkin_make


# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
# add the scene analyser code
# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~

#RUN  mkdir -p /root/scene_analyser
WORKDIR /root/scene_analyser
RUN  git clone git@github.com:usmanzahidi/MaskPredictor.git .
COPY  scene_analyser/docker_extra/setup.py /root/scene_analyser/
RUN  python setup.py develop

RUN  apt-get install ros-melodic-cv-bridge
RUN  pip install rospkg

# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
# switch to final directory and start a terminal
# ~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~
WORKDIR  /root/scene_analyser/src/scene_analyser

CMD /bin/bash
