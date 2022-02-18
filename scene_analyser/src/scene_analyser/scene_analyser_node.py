#!/usr/bin/env python

from __future__ import print_function


import rospy
#import cv2 # opencv python library
#import numpy as np

try:
    import Queue as queue # python 2.7
except:
    import queue

from sensor_msgs.msg import CameraInfo, Image



class SceneAnalyserNode( object ):
    """ This class represents the ROS node for the scene analyser. """
    
    def __init__( self ):
        rospy.init_node( 'scene_analyser', anonymous=False )
        self.topic_prefix = rospy.get_param( '~topic_prefix', '/semantic/' ) # this prefix helps to avoid topic name collisions
        self.topic_rgb = rospy.get_param( '~rgb', 'todo_default' )
        self.topic_depth = rospy.get_param( '~depth', 'todo_default' )
        self.topic_cam_info = rospy.get_param( '~camera_info', 'todo_default' )
        
        self.output_topics = ( # list of topics that we intent to publish
            'canopy',
            'berry',
            'rigid',
            'background')
        
        self.publisher = dict() # dictionary of publishers, topic acts as key
        self.subscribe()
        self.advertise()
    
    def subscribe( self ):
        """ subscribes to the relevant topics """
        # todo: setup callback functions for the subscribers and subscribe to the topics
        rospy.Subscriber(self.topic_rgb, Image, self.callback_rgb )
        rospy.Subscriber(self.topic_depth, Image, self.callback_depth )
        rospy.Subscriber(self.topic_cam_info, Image, self.callback_cam_info )
    
    def advertise( self ):
        """ sets up the publisher for the result images """
        for topic in self.ouput_topics:
            self.publisher[topic] = rospy.Publisher( self.topic_prefix + topic, Image )
    
    def callback_rgb( self, msg ):
        pass
    
    def callback_depth( self, msg ):
        pass
    
    def callback_cam_info( self, msg ):
        pass
    
    def spin( self ):
        """ ros main loop """
        rospy.spin()

