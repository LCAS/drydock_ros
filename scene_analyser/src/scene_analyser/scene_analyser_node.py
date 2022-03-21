#!/usr/bin/env python3

from __future__ import print_function

import time
import threading

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
        self.cam_info_suffix = '/cam_info'
        self.topic_rgb = rospy.get_param( '~rgb', '/rbg' )
        self.topic_depth = rospy.get_param( '~depth', '/depth' )
        self.topic_cam_info = rospy.get_param( '~camera_info', '/cam_info' )
        self.topic_rgb = rospy.get_param( '~rgb', '/camera/saga_arm_d435e/color/image_raw' )
        self.topic_depth = rospy.get_param( '~depth', '/camera/saga_arm_d435e/aligned_depth_to_color/image_raw' )
        self.topic_cam_info = rospy.get_param( '~camera_info', '/camera/saga_arm_d435e/aligned_depth_to_color/camera_info' )
        self.bridge = cv_bridge.CvBridge()
        # @todo: make topic storage a queue to be thread save
        self.msg_rgb = None
        self.msg_depth = None
        self.msg_cam_info = None
        self.model_file  = rospy.get_param( '~model_file', '/root/scene_analyser/model/fp_model.pth' )
        self.config_file = rospy.get_param( '~config_file', 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml' ) # /detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
        self.metadata_file = rospy.get_param( '~metadata_file', '/opt/py3_ws/src/drydock_ros/drydock_ros/scene_analyser/src/MaskPredictor/data/metadata.pkl' )
        self.mask_predictor = None
        self.class_list = [ ClassNames.STRAWBERRY, ClassNames.CANOPY, ClassNames.RIGID_STRUCT, ClassNames.BACKGROUND ]
        new_thread = threading.Thread( target=self.delayed_init )
        new_thread.start()

    
    def delayed_init( self ):
        time.sleep( 0.5 )
        import sys
        import logging
        logging.basicConfig(
            level=logging.DEBUG,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%d/%b/%Y %H:%M:%S",
            stream=sys.stdout)
        print( 'logger modified' )
        # we need to set logging output to rosout, otherwise errors in MaskPredictor will only show up in the log file
        #logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
        logging.getLogger('rosout').setLevel(logging.WARN)
        logging.getLogger('roserr').setLevel(logging.WARN)
        time.sleep( 0.5 )
        logging.error( 'FOOBAR!!!1!' )
        try:
            self.load_mask_predictor()
        except Exception as e:
            print( 'failed to load model:', e )
        
        self.output_topics = ( # list of topics that we intent to publish
            'berry',
            'canopy',
            'rigid',
            'background')
        
        self.publisher_rgbd = dict() # dictionary of publishers, topic acts as key
        self.publisher_cam_info = dict() # dictionary of publishers, image topic acts as key (not CameraInfo topic!)
        print( 'setting up subscriber' )
        self.subscribe()
        print( 'setting up publisher' )
        self.advertise()
        print( 'init complete' )
        
    
    def load_mask_predictor( self ):
        """ loads the mask predictor.
        Note: if this failes the application shuts down silently despite the try block. probably because of a sys.exit call. """
        print( 'loading mask predictor' )
        try:
            time_start = time.time()
            self.mask_predictor = MasksPredictor( self.model_file, self.config_file, self.metadata_file )
            print( 'MaskPredictor loaded, time spend: {:.3f}s'.format(time.time()-time_start) )
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
            self.publisher_rgbd[topic] = rospy.Publisher( self.topic_prefix + topic, Image, queue_size=1 )
        self.publisher_cam_info[topic] = rospy.Publisher( self.topic_prefix + topic + self.cam_info_suffix, CameraInfo, queue_size=1 )
    
    def publish( self, messages, cam_info ):
        """ publishes all messages to the respective topics. expects the messages to be in the same order as self.output_topics """
        for i in range(len(messages)):
            self.publisher_rgbd[self.output_topics[i]].publish( messages[i] )
            self.publisher_cam_info[self.output_topics[i]].publish( cam_info )
    
    def run( self ):
        #if not self.mask_predictor:
        self.load_mask_predictor()
        """ starts the prediction process by converting the ROS messages to OpenCV images and passing the data to the predictor """
        if not self.msg_rgb or not self.msg_depth or not self.msg_cam_info:
            return
        # @todo: prevent possible race condition here, or at least reduce risk
        rgbd_image = self.read_message_data()
        cam_info = self.msg_cam_info
        self.msg_rgb = None
        self.msg_depth = None
        self.msg_cam_info = None
        if rgbd_image is None:
            return
        if not self.mask_predictor: # mask predictor might have been turned off for testing
            print( 'MaskPredictor not loaded yet' )
            return
        print( 'running mask predictor' )
        depth_masks = self.mask_predictor.get_predictions( rgbd_image, self.class_list )
        print( 'masks computed, publishing results' )
        messages = [self.bridge.cv_to_imgmsg(mask, "mono8") for mask in depth_masks ]
        self.publish( messages, cam_info )
    
    def read_message_data( self ):
        """ reads the stored ROS image messages. The messages are converted into OpenCV images and merged into an rgbd image which is then returned.
        returns None on error. """
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2( self.msg_rgb ) # or rgb8
        except Exception as e:
            print( 'failed to convert ROS rgb image to OpenCV', e )
            return None
        try:
            cv_depth = self.bridge.imgmsg_to_cv2( self.msg_depth )
        except Exception as e:
            print( dir(self.msg_depth) )
            print( 'failed to convert ROS depth image to OpenCV', e )
            return None
        
        if cv_rgb.shape[0:2] != cv_depth.shape[0:2]:
            print( 'rgb and depth image size mismatch. rgb={}, depth={}'.format(cv_rgb.shape, cv_depth.shape) )
            return None
        
        rgbd_image  = np.dstack( (cv_rgb, cv_depth[:,:]) )
        return rgbd_image
    
    def callback_rgb( self, msg ):
        self.msg_rgb = msg
        print( 'rgb callback ({}x{}, {})'.format(msg.width, msg.height, msg.encoding) )
        self.run()
    
    def callback_depth( self, msg ):
        self.msg_depth = msg
        print( 'depth callback ({}x{}, {})'.format(msg.width, msg.height, msg.encoding) )
        self.run()
    
    def callback_cam_info( self, msg ):
        self.msg_cam_info = msg
    
    def spin( self ):
        """ ros main loop """
        print( 'spinning' )
        rospy.spin()

