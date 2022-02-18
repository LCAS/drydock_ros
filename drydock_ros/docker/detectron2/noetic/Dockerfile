## Basis for running detectron2 backend with ros in a container.
## Extended from the nvcr.io/nvidia/pytorch:21.02-py3 image.
##
## Prerequisites:
##   - docker
##   - nvidia driver
##   - nvidia container toolkit
##
## Build with:
##   docker build -t noetic_detectron2:local .
##
## Save with:
##   docker save noetic_detectron2:local | gzip > noetic_detectron2.tar.gz
##
## Run with:
##   docker run --network host --gpus all --name noetic_detectron2_backend --entrypoint /bin/bash --rm -it noetic_detectron2:local
##
## Commit With:
##  docker commit noetic_detectron2_backend noetic_detectron2:local 

FROM  nvcr.io/nvidia/pytorch:21.02-py3

## Meta information
LABEL detectron2.version="0.5" maintainers="Robert Belshaw <rbelshaw@sagarobotics.com>"

## Install ROS
ENV ROS_DISTRO noetic
ENV TZ Europe/London
RUN DEBIAN_FRONTEND=noninteractive apt update --no-install-recommends && \
    DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common lsb-release curl apt-transport-https git --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* && \
    echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && \
    ## https://discourse.ros.org/t/ros-gpg-key-expiration-incident/20669
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    DEBIAN_FRONTEND=noninteractive apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y ros-${ROS_DISTRO}-ros-base python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential --no-install-recommends && \ 
    DEBIAN_FRONTEND=noninteractive apt install -y build-essential && \
    DEBIAN_FRONTEND=noninteractive apt install -y python3-rosdep --no-install-recommends && rosdep init && rosdep update

#CV2 Dependencies 
RUN DEBIAN_FRONTEND=noninteractive apt install -y ffmpeg libsm6 libxext6 

WORKDIR /

## Install dependencies. See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --no-cache-dir opencv-python && \
    # At time of writing CUDA 10.2 is the native torch version but we need 11.1 for RTX3000
    # pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir 'git+https://github.com/facebookresearch/fvcore' && \
    # Clone detectron2
    git clone https://github.com/facebookresearch/detectron2 && \
    # Install detectron2
    python -m pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html && \
    pip install --no-cache-dir rospkg

# Docker clean-up
RUN rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash", "-c"]