#!/usr/bin/env python3

from __future__ import print_function

import time
#import logging

import numpy as np
import cv_bridge
#import rospy

#try:
#    from Queue import Queue # python 2.7
#except:
#    from queue import Queue # python 3+

#from sensor_msgs.msg import CameraInfo, Image

from MaskPredictor.masks_predictor import MasksPredictor, ClassNames, OutputType




class ROSMaskPredictor( object ):
    """ this class takes ros image messages and runs the MasksPredictor on the data. convertion between ros messages and opencv image objects is done here as well. """
    
    def __init__( self, model_file, config_file, metadata_file, num_classes=3 ):
        """ note: model_file and config_file are strings, not file objects. """
        self.bridge = cv_bridge.CvBridge()
        self.class_list = [ ClassNames.STRAWBERRY, ClassNames.CANOPY, ClassNames.RIGID_STRUCT, ClassNames.BACKGROUND ]
        self.num_classes = num_classes
        self.mask_predictor = self.load_mask_predictor( model_file, config_file, metadata_file, num_classes )
    
    def predict( self, msg_img_rgb, msg_img_depth, msg_cam_info=None ):
        """ runs the mask predictor on the provided data. returns a list of depth images, one for each category/label.
        both input and output images are ros Image messages. conversion to cv2 images is done inside this class.
        
        returns: list of ros Image messages """
        rgbd_image = self.msg_to_cvimage( msg_img_rgb, msg_img_depth, msg_cam_info )
        depth_masks = self.mask_predictor.get_predictions( rgbd_image, self.class_list, OutputType.DEPTH_MASKS )
        ros_images = []
        for c in range( len(depth_masks) ):
            mono_img = depth_masks[:,:,c]
            mono_img = mono_img.astype(np.uint16)
            mask_img_msg = self.bridge.cv2_to_imgmsg( mono_img )
            self.copy_msg_header( msg_img_depth, mask_img_msg )
            ros_images.append( mask_img_msg )
        return ros_images
    
    def copy_msg_header( self, source, dest ):
        dest.header.stamp = source.header.stamp
        dest.header.frame_id = source.header.frame_id
    
    def load_mask_predictor( self, model_file, config_file, metadata_file, num_classes ):
        """ loads the mask predictor.
        Note: if this failes the application may shut down silently despite the try block. this is caused by sys.exit calls inside MasksPredictor. """
        print( 'loading MaskPredictor' )
        try:
            time_start = time.time()
            mask_predictor = MasksPredictor( model_file, config_file, metadata_file, num_classes )
            print( 'MaskPredictor loaded, time spend: {:.3f}s'.format(time.time()-time_start) )
            return mask_predictor
        except Exception as e:
            print( 'failed to load mask predictor', e )
        return None
    
    def depth_masks_to_ros_image( self, depth_masks ):
        """ unused atm """
        yellow = depth_masks[:, :, 0].copy()
        green = depth_masks[:, :, 1].copy()
        red = depth_masks[:, :, 2].copy()
        blue = depth_masks[:, :, 3].copy()
        
        bgr_image = depth_masks[:, :, 0:3].copy()
        bgr_image[:, :, 0] = blue
        bgr_image[:, :, 1] = green + yellow
        bgr_image[:, :, 2] = red + yellow
        return self.bridge.cv2_to_imgmsg( bgr_image )
    
    def msg_to_cvimage( self, msg_img_rgb, msg_img_depth, msg_cam_info ):
        """ reads the ROS image messages and converts them into a merged OpenCV rgbd image.
        returns None on error. """
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2( msg_img_rgb )
        except Exception as e:
            print( 'failed to convert ROS rgb image to OpenCV', e )
            return None
        try:
            cv_depth = self.bridge.imgmsg_to_cv2( msg_img_depth )
            cv_depth = self.bridge.imgmsg_to_cv2( msg_img_depth, '32FC1' )
            #cv_depth = cv_depth.astype( 'float32' )
        except:
            print( 'failed to convert ROS depth image to OpenCV' )
            return None
        
        if cv_rgb.shape[0:2] != cv_depth.shape[0:2]:
            print( 'rgb and depth image size mismatch. rgb={}, depth={}'.format(cv_rgb.shape, cv_depth.shape) )
            return None
        
        rgbd_image = np.dstack( (cv_rgb, cv_depth[:,:]) )
        return rgbd_image







