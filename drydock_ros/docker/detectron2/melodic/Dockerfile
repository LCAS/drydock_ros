## Basis for running detectron2 backend with ros in a container.
## Extended from the nvcr.io/nvidia/pytorch:20.11-py3 image.
##
## Prerequisites:
##   - docker
##   - nvidia driver
##   - nvidia container toolkit
##
## Build with:
##   docker build -t melodic_detectron2:local .
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
    pip install --no-cache-dir opencv-python && \
    # At time of writing CUDA 10.2 is the native torch version but we need 11.1 for RTX3000
    # pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir 'git+https://github.com/facebookresearch/fvcore' && \
    # Clone detectron2
    git clone https://github.com/facebookresearch/detectron2 && \
    # Install detectron2
    python -m pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html && \
    pip install --no-cache-dir rospkg

RUN rosdep update

### MQTT
RUN source /opt/ros/${ROS_DISTRO}/setup.sh && mkdir -p /opt/fastpick/src && cd /opt/fastpick/src && catkin_init_workspace . \
    && git clone https://github.com/LCAS/mqtt_bridge.git 

RUN apt-get install -y mosquitto-clients mosquitto

COPY . /opt/fastpick/src/drydock_ros    
RUN DEBIAN_FRONTEND=noninteractive apt update --no-install-recommends

# hack required to use Python2 here, change PATH to put /usr/bin first again...
RUN source /opt/ros/${ROS_DISTRO}/setup.sh && cd /opt/fastpick/src && rosdep install -y -i --from-paths . && cd .. && PATH=/usr/bin:$PATH catkin_make install

# Docker clean-up
RUN rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash", "-c", "PATH=/usr/bin:$PATH; source /opt/fastpick/install/setup.bash; roslaunch drydock_ros fastpick-server.launch"]

# build this e.g. from the top level opf repo as 