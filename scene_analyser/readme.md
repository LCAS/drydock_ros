# How to build and run the scene analyser


### 0) clone git repository

    mkdir ~/github_drydock
    cd ~/github_drydock
    git clone --recurse-submodules https://github.com/LCAS/drydock_ros.git

### 1) build docker image(s)

    # docker-compose commands must be executed in the same directory as the yaml file
    cd ~/github_drydock/drydock_ros/drydock_ros/docker
    # to build all images:
    docker-compose build
    # or to build a single image:
    docker-compose build semantic_segmentation

note: to update the MaskPredictor submodule, go to the submodule folder and run git pull:

    cd ~/github_drydock/drydock_ros/scene_analyser/src/MaskPredictor
    git pull origin main

### 2) run the images

    # docker-compose commands must be executed in the same directory as the yaml file
    cd ~/github_drydock/drydock_ros/drydock_ros/docker
    docker-compose up mqtt_client
    # or if you need an interactive console (e.g. testing & debugging)
    docker-compose run johann_testing


Note: more instructions about how to run the action-server/client coming soon
