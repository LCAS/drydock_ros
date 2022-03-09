#!/usr/bin/env python3

from __future__ import print_function


import numpy as np
import cv_bridge
import rospy

#try:
#    from Queue import Queue # python 2.7
#except:
#    from queue import Queue # python 3+

from sensor_msgs.msg import CameraInfo, Image

from MaskPredictor.masks_predictor import MasksPredictor, ClassNames


class SceneAnalyserNode( object ):
    """ This class represents the ROS node for the scene analyser. """
    
    def __init__( self ):
        rospy.init_node( 'scene_analyser', anonymous=False )
        print( 'SceneAnalyserNode.__init__' )
        self.topic_prefix = rospy.get_param( '~topic_prefix', '/semantic/' ) # this prefix helps to avoid topic name collisions
        self.topic_rgb = rospy.get_param( '~rgb', 'todo_default' )
        self.topic_depth = rospy.get_param( '~depth', 'todo_default' )
        self.topic_cam_info = rospy.get_param( '~camera_info', 'todo_default' )
        self.bridge = cv_bridge.CvBridge()
        self.msg_rgb = None
        self.msg_depth = None
        self.msg_cam_info = None
        self.model_file  = 'root/scene_analyser/model/fp_ss_model.pth'
        self.config_file = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
        self.mask_predictor = None
        self.class_list = [ ClassNames.STRAWBERRY, ClassNames.CANOPY, ClassNames.RIGID_STRUCT, ClassNames.BACKGROUND ]
        try:
            self.load_mask_predictor()
        except Exception as e:
            print( 'failed to load model:', e )
        
        self.output_topics = ( # list of topics that we intent to publish
            'berry',
            'canopy',
            'rigid',
            'background')
        
        self.publisher = dict() # dictionary of publishers, topic acts as key
        print( 'setting up subscriber' )
        self.subscribe()
        print( 'setting up publisher' )
        self.advertise()
        print( 'init complete' )
    
    def load_mask_predictor( self ):
        """ loads the mask predictor.
        Note: if this failes the application shuts down silently despite the try block. probably because of a sys.exit call. """
        return
        print( 'laoding model' )
        try:
            self.mask_predictor = MasksPredictor( self.model_file, self.config_file )
        except Exception as e:
            print( 'failed to load mask predictor', e )
    
    def subscribe( self ):
        """ subscribes to the relevant topics """
        print( 'subscribing to topics:' )
        for t in (self.topic_rgb, self.topic_depth, self.topic_cam_info ):
            print( '  - "{}"'.format(t) )
        rospy.Subscriber( self.topic_rgb, Image, self.callback_rgb )
        rospy.Subscriber( self.topic_depth, Image, self.callback_depth )
        rospy.Subscriber( self.topic_cam_info, CameraInfo, self.callback_cam_info )
    
    def advertise( self ):
        """ sets up the ROS publisher for the result images """
        print( 'advertising topics:' )
        for topic in self.output_topics:
            print( '  - "{}"'.format(self.topic_prefix+topic) )
            self.publisher[topic] = rospy.Publisher( self.topic_prefix + topic, Image )
    
    def publish( self, messages ):
        """ publishes all messages to the respective topics. expects the messages to be in the same order as self.output_topics """
        for i in range(len(messages)):
            self.publisher[self.output_topics[i]].publish( messages[i] )
    
    def run( self ):
        """ starts the prediction process by converting the ROS messages to OpenCV images and passing the data to the predictor """
        if not self.msg_rgb or not self.msg_depth:
            return
        rgbd_image = self.read_message_data()
        if not rgbd_image:
            return
        depth_masks = self.mask_predictor.get_predictions( rgbd_image, self.class_list )
        messages = [self.bridge.cv_to_imgmsg(mask, "mono8") for mask in depth_masks ]
        self.publish( messages )
        self.msg_rgb = None
        self.msg_depth = None
    
    def read_message_data( self ):
        """ reads the stored ROS image messages. The messages are converted into OpenCV images and merged into an rgbd image which is then returned.
        returns None on error. """
        try:
            cv_rgb = self.bridge.imgmsg_to_cv(self.msg_rgb, "bgr8") # or rgb8
            if not cv_rgb:
                raise Exception('cv_rgb error')
        except:
            print( 'failed to convert ROS rgb image to OpenCV' )
            return None
        try:
            cv_depth = self.bridge.imgmsg_to_cv(self.msg_depth, "mono8") # alternative: mono16
            if not cv_depth:
                raise Exception('cv_depth error')
        except:
            print( 'failed to convert ROS depth image to OpenCV' )
            return None
        
        if cv_rgb.shape != cv_depth.shape:
            print( 'rgb and depth image size mismatch. rgb={}, depth={}'.format(cv_rgb.shape, cv_depth.shape) )
            return None
        
        rgbd_image  = np.dstack( (cv_rgb, cv_depth[:,:,0]) )
        return rgbd_image
    
    def callback_rgb( self, msg ):
        self.msg_rgb = msg
    
    def callback_depth( self, msg ):
        self.msg_depth = msg
    
    def callback_cam_info( self, msg ):
        self.msg_cam_info = msg
    
    def spin( self ):
        """ ros main loop """
        print( 'spinning' )
        rospy.spin()

