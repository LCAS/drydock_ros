#!/usr/bin/env python3

from __future__ import print_function

#import time

import rospy
from sensor_msgs.msg import CameraInfo, Image



class SceneAnalyserActionClient( object ):
    def __init__( self ):
        rospy.init_node( 'scene_analyser_action_client' )
        self.topic_rgb = rospy.get_param( '~topic_rgb', '/rbg' )
        self.topic_depth = rospy.get_param( '~topic_depth', '/depth' )
        self.topic_cam_info = rospy.get_param( '~topic_cam_info', '/cam_info' )
        self.subscribe()
    
    def subscribe( self ):
        """ subscribes to the relevant topics """
        rospy.Subscriber( self.topic_rgb, Image, self.callback_rgb )
        rospy.Subscriber( self.topic_depth, Image, self.callback_depth )
        rospy.Subscriber( self.topic_cam_info, CameraInfo, self.callback_cam_info )
    
    def callback_rgb( self, msg ):
        pass
    
    def callback_depth( self, msg ):
        pass
    
    def callback_cam_info( self, msg ):
        pass
    
    def spin( self ):
        rospy.spin()





if __name__ == '__main__':

    client = SceneAnalyserActionClient()
    client.spin()