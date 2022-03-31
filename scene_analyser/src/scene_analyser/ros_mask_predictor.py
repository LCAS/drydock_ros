#!/usr/bin/env python3

from __future__ import print_function

import time

import numpy as np
import cv_bridge
#import rospy
#from sensor_msgs.msg import CameraInfo, Image

from MaskPredictor.masks_predictor import MasksPredictor, ClassNames, OutputType




class ROSMaskPredictor( object ):
    """ this class takes ros image messages and runs the MasksPredictor on the data. convertion between ros messages and opencv image objects is done here as well. """
    
    def __init__( self, model_file, config_file, metadata_file, num_classes=3 ):
        """ note: model_file and config_file are strings, not file objects. """
        self.bridge = cv_bridge.CvBridge()
        self.class_list = [ ClassNames.STRAWBERRY, ClassNames.CANOPY, ClassNames.RIGID_STRUCT, ClassNames.BACKGROUND ]
        self.class_labels = [ ClassNames(c).name for c in self.class_list ]
        self.num_classes = num_classes
        self.mask_predictor = self.load_mask_predictor( model_file, config_file, metadata_file, num_classes )
    
    def predict( self, msg_img_rgb, msg_img_depth, msg_cam_info=None ):
        """ runs the mask predictor on the provided data. returns a list of depth images, one for each category/label.
        both input and output images are ros Image messages. conversion to cv2 images is done inside this class.
        
        returns:
            ros_depth_images: list of ros Image messages. same as the input depth image, but every pixel that is not part of the label is set to zero
            ros_rgb_image: same deal as above, but as rgb images
            class_label: list of string of the class labels. same order as the rest of the lists here
            label_image: a ros image message that contains color-coded classes """
        rgbd_image = self.msg_to_cvimage( msg_img_rgb, msg_img_depth, msg_cam_info )
        depth_masks, unused = self.mask_predictor.get_predictions( rgbd_image, self.class_list, OutputType.DEPTH_MASKS, ClassNames.ALL )
        unused, rgb_masks = self.mask_predictor.get_predictions( rgbd_image, self.class_list, OutputType.COLOR_MASKS, ClassNames.ALL )
        ros_depth_images = []
        ros_rgb_images = []
        for c in range(4):
            # depth images
            mono_img = depth_masks[:,:,c]
            mono_img = mono_img.astype(np.uint16)
            mask_img_msg = self.bridge.cv2_to_imgmsg( mono_img )
            self.copy_msg_header( msg_img_depth, mask_img_msg )
            ros_depth_images.append( mask_img_msg )
            # rgb images:
            rgb_img = rgb_masks[c]
            rgb_img = rgb_img.astype(np.uint8)
            mask_img_msg = self.bridge.cv2_to_imgmsg( rgb_img, 'rgb8' )
            self.copy_msg_header( msg_img_depth, mask_img_msg )
            ros_rgb_images.append( mask_img_msg )
        label_image = self.depth_masks_to_ros_image( depth_masks )
        #print( 'label_image={}'.format(str(label_image)[:500]) )
        print( 'finished predictions' )
        return ros_depth_images, ros_rgb_images, self.class_labels, label_image
    
    def copy_msg_header( self, source, dest ):
        """ copies the time stamp and frame_id field from source to dest """
        dest.header.stamp = source.header.stamp
        dest.header.frame_id = source.header.frame_id
    
    def load_mask_predictor( self, model_file, config_file, metadata_file, num_classes ):
        """ loads the mask predictor.
        Note: if this failes the application may shut down silently despite the try block. this is caused by sys.exit calls inside MasksPredictor. """
        print( 'loading MaskPredictor, labels={}'.format(self.class_labels) )
        try:
            time_start = time.time()
            mask_predictor = MasksPredictor( model_file, config_file, metadata_file, num_classes )
            print( 'MaskPredictor loaded, time spend: {:.3f}s'.format(time.time()-time_start) )
            return mask_predictor
        except Exception as e:
            print( 'failed to load mask predictor', e )
        return None
    
    def depth_masks_to_ros_image( self, depth_masks ):
        """ creates a color-coded label image (ros image message). every pixel has one of four colors, based on the associated label that applies to that particualr pixel """
        yellow = depth_masks[:, :, 0].copy()
        green = depth_masks[:, :, 1].copy()
        red = depth_masks[:, :, 2].copy()
        blue = depth_masks[:, :, 3].copy()
        
        bgr_image = depth_masks[:, :, 0:3].copy()
        bgr_image[:, :, 0] = blue
        bgr_image[:, :, 1] = green + yellow
        bgr_image[:, :, 2] = red + yellow
        return self.bridge.cv2_to_imgmsg( bgr_image.astype(np.uint8), 'bgr8' )
    
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







