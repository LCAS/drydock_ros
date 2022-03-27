## Scene Analyser Components

### Scene Analyser Action Server

The Scene Analyser Action Server runs a ROS node utilizing the actionlib interface to provide an action server. The action server listens for goal messages at /scene_analyser/goal and publishes results to /scene_analyser/result. The feedback topic is currently ignored. The request contains a RGB image, a (synced) depth image, and the CameraInfo message for the depth image. The result is an array of depth images, one for each segmentation label. The action message definition can be found here: [semantic_segmentation.action](https://github.com/LCAS/drydock_ros/blob/main/scene_analyser/action/semantic_segmentation.action).


#### How to start the action server:

    rosrun scene_analyser sa_action_server.py

Note, that the action server is also launched when the launch file es executed, which should happen automatically if the docker image is run without an interactive console.

#### Action Server Parameters:

- **model_file**: path to the pytorch model, default is *"/root/scene_analyser/model/fp_model.pth"*
- **config_file**: path tp the detectron2 config file, default is *"COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"*
- **metadata_file**: path to the MaskPredictor metadata file, default is *"/opt/py3_ws/src/drydock_ros/scene_analyser/src/MaskPredictor/data/metadata.pkl"*
- **num_classes**: number of classes for the mask predictor. must match the model. Default value is *"3"*, which goes with the default model. A custom model for plastic test plants might work with only 2 classes. Nte, that the resulting depth maskes are N+1, the last depth image containing all valid points that are not labeled.

All default values should work out of the box


### Scene Analyser Action Client

The action client listens to camera topics (rgb, depth, & CameraInfo) and tries to find synchronized topics. If successful, an action goal is send to the action server for further processing. This process is blocking until the result from the action server is returned.

#### How to start the action client:

    rosrun scene_analayser sa_action_client.py

#### Action Client Parameters:

- **topic_rgb**: topic name of the rgb image, default is *"/camera/saga_arm_d435e/color/image_raw"*
- **topic_depth**: topic name for the depth image, default is *"/camera/saga_arm_d435e/aligned_depth_to_color/image_raw"*
- **topic_cam_info**: topic name for the depth image's camera info, default is *"/camera/saga_arm_d435e/aligned_depth_to_color/camera_info"*


## How to build and run the scene analyser


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
