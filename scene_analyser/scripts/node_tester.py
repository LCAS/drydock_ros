#!/usr/bin/env python3
""" scene analyser node_tester
this script reads a random rgb/depth image pair from the provided folder and publishers it. designed for testing the scene_analyser_node.py script.
Note: we do not publish any cam_info topic, just the two images.
"""
from __future__ import print_function


import os
import random

import cv2
import cv_bridge

import rospy
from sensor_msgs.msg import Image



class TesterNode( object ):
    def __init__( self ):
        rospy.init_node( 'SceneAnalyserTester' )
        self.src_folder = '/opt/py3_ws/src/drydock_ros/drydock_ros/scene_analyser/src/MaskPredictor/images/'
        self.rgb_topic = '/rgb'
        self.depth_topic = '/depth'
        self.cv_bridge = cv_bridge.CvBridge()
        self.rgb_publisher = rospy.Publisher( self.rgb_topic, Image, queue_size=0 )
        self.depth_publisher = rospy.Publisher( self.depth_topic, Image, queue_size=0 )

    def load_data( self ):
        folder_rgb = self.src_folder + 'rgb/'
        folder_depth = self.src_folder + 'depth/'
        try:
            everything = os.listdir( folder_rgb )
            rgb_files = [ name for name in everything if name.endswith('.png') ]
            filename = random.choice( rgb_files )
            print( 'reading file {}'.format(folder_rgb + filename) )
            rgb_image = cv2.imread( folder_rgb + filename, cv2.IMREAD_UNCHANGED )
            print( 'image shape={}'.format(rgb_image.shape) )
            print( 'image data type={}'.format(rgb_image.dtype) )
            ros_image = self.cv_bridge.cv2_to_imgmsg( rgb_image, encoding='bgr8' )
            print( 'publishing' )
            self.rgb_publisher.publish( ros_image )
        except Exception as e:
            print( 'failed to read rgb image {}'.format(folder_rgb+filename), e )
        try:
            print( 'reading file {}'.format(folder_depth + filename) )
            depth_image = cv2.imread( folder_depth + filename, cv2.IMREAD_UNCHANGED )
            print( 'image shape={}'.format(depth_image.shape) )
            print( 'image data type={}'.format(depth_image.dtype) )
            ros_image = self.cv_bridge.cv2_to_imgmsg( depth_image, encoding='mono16' )
            print( 'publishing' )
            self.rgb_publisher.publish( ros_image )
        except Exception as e:
            print( 'failed to read depth image {}'.format(folder_depth+filename), e )
    
    def spin( self ):
        """ ros main loop """
        rospy.spin()


if __name__ == '__main__':

    tester = TesterNode()
    tester.load_data()
    tester.spin()
