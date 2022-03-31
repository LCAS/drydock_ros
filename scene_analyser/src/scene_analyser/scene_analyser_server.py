#import threading

import rospy
import actionlib
from sensor_msgs.msg import Image

import scene_analyser.msg as action_msgs
from scene_analyser.ros_mask_predictor import ROSMaskPredictor


class SceneAnalyserActionServer( object ):
    def __init__( self, action_name='/scene_analyser' ):
        rospy.init_node('scene_analyser_action_server')
        self.model_file  = rospy.get_param( '~model_file', '/root/scene_analyser/model/fp_model.pth' )
        self.config_file = rospy.get_param( '~config_file', 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml' ) # /detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
        self.metadata_file = rospy.get_param( '~metadata_file', '/opt/py3_ws/src/drydock_ros/scene_analyser/src/MaskPredictor/data/metadata.pkl' )
        self.num_classes = int( rospy.get_param( '~num_classes', '3' ) )
        self.mask_predictor = ROSMaskPredictor( self.model_file, self.config_file, self.metadata_file, self.num_classes )
        self.publish_mask_images = rospy.get_param( '~publish_masks', 'True' ) # if true, we additionally publish the masks as individual topics
        self.mask_img_publishers = {}
        self.mask_img_topic_prefix = '/masks/'
        self.action_name = action_name
        self.action_server = actionlib.SimpleActionServer(
            self.action_name,
            action_msgs.semantic_segmentationAction,
            execute_cb=self.execute,
            auto_start = False)
        self.action_server.start()
        print( 'Scene Analyser Action Server is ready' )
        
    def advertise( self, mask_id=0 ):
        """ setups the publisher for the provided mask_id. """
        self.mask_img_publishers[mask_id] = rospy.Publisher( self.mask_img_topic_prefix + str(mask_id), Image, queue_size=1 )
    
    def publish( self, img_msg, mask_id ):
        """ calls the ros publisher according to the provided mask_id """
        if not mask_id in self.mask_img_publishers:
            self.advertise( mask_id )
        self.mask_img_publishers[mask_id].publish( img_msg )
    
    def execute( self, goal ):
        """ executed when the action server receives a goal """
        print( 'scene analyser action server - goal received' )
        img_rgb = goal.rgb
        img_depth = goal.depth
        cam_info = goal.cam_info
        depth_images, rgb_images, labels, label_image = self.mask_predictor.predict( img_rgb, img_depth, cam_info )
        """result = action_msgs.semantic_segmentationActionResult()
        print( 'action result={}'.format(result) )
        print( dir(result) )
        result.header.stamp = goal.header.stamp # result shares the same time stamp as the goal, to make it easier to match the two
        result.result.depth = masks
        print( 'sending action result' )
        """
        result = action_msgs.semantic_segmentationResult()
        result.depth = depth_images
        result.labels = labels
        result.cam_info = cam_info
        self.action_server.set_succeeded( result=result )
        if self.publish_mask_images:
            for i in range(len(depth_images)):
                self.publish( depth_images[i], 'depth/{}'.format(i) )
                self.publish( rgb_images[i], 'rgb/{}'.format(i) )
            self.publish( label_image, 'labels' )
    
    def spin( self ):
        rospy.spin()
