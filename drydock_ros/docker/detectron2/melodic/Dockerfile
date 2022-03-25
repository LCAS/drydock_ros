## Basis for running detectron2 backend with ros in a container.
## Extended from the nvcr.io/nvidia/pytorch:20.11-py3 image.
##
## Prerequisites:
##   - docker
##   - nvidia driver
##   - nvidia container toolkit
##
## Build with:
##   docker build -t melodic_detectron2:local --no-cache -f drydock_ros/drydock_ros/docker/detectron2/melodic/Dockerfile .
##
## Save with:
##   docker save melodic_detectron2:local | gzip > melodic_detectron2.tar.gz
##
## Run with:
##   docker run --network host --gpus all --name melodic_detectron2_backend --entrypoint /bin/bash --rm -it melodic_detectron2:local
##
## Commit With:
##  docker commit melodic_detectron2_backend melodic_detectron2:local 

FROM  nvcr.io/nvidia/pytorch:20.11-py3

## Meta information
LABEL detectron2.version="0.5" maintainers="Robert Belshaw <rbelshaw@sagarobotics.com>"

## Install ROS
ENV ROS_DISTRO melodic
ENV TZ Europe/London
RUN DEBIAN_FRONTEND=noninteractive apt update --no-install-recommends && \
    DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common lsb-release curl apt-transport-https git --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* && \
    echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && \
    ## https://discourse.ros.org/t/ros-gpg-key-expiration-incident/20669
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    DEBIAN_FRONTEND=noninteractive apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y ros-${ROS_DISTRO}-catkin --no-install-recommends && \
    DEBIAN_FRONTEND=noninteractive apt install -y ros-${ROS_DISTRO}-ros-base python-catkin-tools python3-venv python3-pip --no-install-recommends && \ 
    DEBIAN_FRONTEND=noninteractive apt install -y build-essential && \
    DEBIAN_FRONTEND=noninteractive apt install -y python-rosdep --no-install-recommends && rosdep init && rosdep update

WORKDIR /

## Install backend in virtual Python3 environment
## Install dependencies. See https://pytorch.org/ for other options if you use a different version of CUDA
RUN python3 -m venv detectron2_venv --clear --system-site-packages && \
    . detectron2_venv/bin/activate && \
    pip install --upgrade pip wheel setuptools && \
    pip install --no-cache-dir absl-py pytest scipy sphinx_glpi_theme && \
    pip install --no-cache-dir opencv-python && \
    # At time of writing CUDA 10.2 is the native torch version but we need 11.1 for RTX3000
    # pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir 'git+https://github.com/facebookresearch/fvcore' && \
    # Clone detectron2
    git clone --depth 1 -b v0.6 --single-branch https://github.com/facebookresearch/detectron2 && \
    # Install detectron2
    python -m pip install -e detectron2 && \
    #python -m pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html && \
    #python -m pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html && \
    pip install --no-cache-dir rospkg

RUN rosdep update


# install mosquitto and tmux
RUN apt-get install -y mosquitto-clients mosquitto tmux

# setup conda packages for ROS
RUN pip3 install tmule rospkg && conda install -c conda-forge ros-catkin setuptools
RUN PATH=/usr/bin:$PATH pip install -r https://raw.githubusercontent.com/LCAS/mqtt_bridge/melodic-devel/requirements.txt 
RUN conda init bash

# install python2 workspace
RUN DEBIAN_FRONTEND=noninteractive apt update --no-install-recommends
RUN source /opt/ros/${ROS_DISTRO}/setup.sh && mkdir -p /opt/py2_ws/src && cd /opt/py2_ws/src && catkin_init_workspace \ 
        && git clone --depth 1 -b melodic-devel https://github.com/LCAS/mqtt_bridge.git \
        && rosdep install -y -i --from-paths . \
        && PATH=/usr/bin:$PATH pip install -r mqtt_bridge/requirements.txt \
        && cd .. \
        && ROS_PYTHON_VERSION=2 PATH=/usr/bin:$PATH catkin_make install

RUN source /opt/ros/${ROS_DISTRO}/setup.sh && mkdir -p /opt/py3_ws/src && cd /opt/py3_ws/src && catkin_init_workspace 

# install virtual desktop
RUN apt-get install -y xfce4-session xfce4-terminal xfce4-panel tigervnc-standalone-server net-tools
RUN mkdir -p /usr/local/lib && git -C /usr/local/lib clone --recursive https://github.com/marc-hanheide/remote_manager.git 
RUN mkdir -p /usr/local/bin && install /usr/local/lib/remote_manager/network-scripts/bin/ngrok1.lnx /usr/local/bin/ngrok1


COPY . /opt/py3_ws/src/drydock_ros    
RUN DEBIAN_FRONTEND=noninteractive apt update --no-install-recommends

RUN source /opt/ros/${ROS_DISTRO}/setup.sh \
    && cd /opt/py3_ws && rosdep install -y -i --from-paths src \
    && ROS_PYTHON_VERSION=3 catkin_make install -DSETUPTOOLS_DEB_LAYOUT=OFF
    #&& ROS_PYTHON_VERSION=2 PATH=/usr/bin:$PATH catkin_make install --pkg mqtt_bridge  
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc; echo "source /opt/py3_ws/install/setup.bash" >> ~/.bashrc

# Docker clean-up
RUN rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash", "-c", "(/usr/local/lib/remote_manager/vnc_runner/noVNC/utils/launch.sh --vnc localhost:5901 --listen 6080 &); rm -f /tmp/.X1-lock /tmp/.X11-unix/X1; (vncserver -localhost yes -depth 24 -geometry 1280x720 -SecurityTypes None -fg :1 &);    ((source /detectron2_venv/bin/activate; source /opt/ros/melodic/setup.bash; source /opt/py3_ws/install/setup.bash; ROS_PYTHON_VERSION=3 roslaunch scene_analyser scene_analyser.launch)&) ; sleep 5; (conda deactivate; export PATH=/usr/bin:$PATH; source /opt/ros/melodic/setup.bash; source /opt/py3_ws/install/setup.bash; source /opt/py2_ws/install/setup.bash; ROS_PYTHON_VERSION=3 roslaunch /opt/py3_ws/install/share/drydock_ros/launch/host.launch)"]

# build this e.g. from the top level opf repo as 
