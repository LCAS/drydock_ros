#!/usr/bin/env python3

from __future__ import print_function

#import time
from threading import Thread, Lock

import scene_analyser.msg as action_msgs

import rospy
import actionlib
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger, TriggerResponse

class SceneAnalyserActionClient( object ):
    """ The SceneAnalyserActionClient is a node that listend to rgb/depth/cameraInfo messages
    and sends action goals to the action server whenever synchronized rgb/depth images are received. """
    def __init__( self ):
        rospy.init_node( 'scene_analyser_action_client' )
        # default values work with the demo bagfile
        self.topic_rgb = rospy.get_param( '~topic_rgb', '/camera/saga_arm_d435e/color/image_raw' )
        self.topic_depth = rospy.get_param( '~topic_depth', '/camera/saga_arm_d435e/aligned_depth_to_color/image_raw' )
        self.topic_cam_info = rospy.get_param( '~topic_cam_info', '/camera/saga_arm_d435e/aligned_depth_to_color/camera_info' )
        self.run_on_service = rospy.get_param( '~run_on_service', True ) 
        self.check_msg_age = rospy.get_param( '~check_msg_age', True ) 
        self.msg_lock = Lock() # to lock access to the messages while reading/writing messages
        self.action_lock = Lock()
        self.msg_rgb = None
        self.msg_depth = None
        self.msg_cam_info = None
        self.use_multithreading = False
        self.time_tolerance = rospy.Duration( 0.1 ) # a time stamp difference greater than this between rgb and depth image will cause the older message to be discarded
        self.action_timeout = rospy.Duration( 10.0 ) # timeout Before the client cancels the goal
        self.subscribe()
        self.action_name = '/scene_analyser'
        self.action_client = actionlib.SimpleActionClient( self.action_name, action_msgs.semantic_segmentationAction )
        if self.run_on_service:
            self.service_server = rospy.Service( self.action_name + "/trigger" , Trigger, self._trigger_service_cb )
        print( 'Scene Analyser Action Client Is Ready' )
    
    def subscribe( self ):
        """ subscribes to the relevant topics """
        print( 'Subscribing To Topics:' )
        print( '  - "{}"'.format( self.topic_rgb) )
        rospy.Subscriber( self.topic_rgb, Image, self.callback_rgb )
        print( '  - "{}"'.format(self.topic_depth) )
        rospy.Subscriber( self.topic_depth, Image, self.callback_depth )
        print( '  - "{}"'.format(self.topic_cam_info) )
        rospy.Subscriber( self.topic_cam_info, CameraInfo, self.callback_cam_info )
    
    def msg_time( self, msg ):
        """ returns the ros Time instance of the provided message """
        stamp = msg.header.stamp
        return rospy.Time( stamp.secs, stamp.nsecs )

    def msg_old(self , msg, timeout=3):
        """ checks the ros Time of the provided messge is not outdated """
        if not self.check_msg_age:
            return True
        stamp = msg.header.stamp
        current = rospy.Time.now()
        diff = current.secs - stamp.secs
        if diff > timeout:
            return False 
        else:
            return True

    def check_msgs( self ):
        """ checks if we have two messages (rgb & depth) whose time stamps are close enough to be considered synchronized. if yes, we proceed with processing the messages further, i.e. sending an action goal """
        if not self.msg_rgb  or  not self.msg_depth  or  not self.msg_cam_info:
            print("Client Error: Camera Msgs Not Recieved")
            return False
        time_rgb = self.msg_time( self.msg_rgb )
        time_depth = self.msg_time( self.msg_depth )
        time_delta = abs( time_rgb - time_depth )
        if not self.msg_old(self.msg_rgb):
            print("Client Error: Camera Msgs Outdated")
            return False
        if time_delta > self.time_tolerance:
            # if the time difference is too large we drop the older message:
            if time_rgb > time_depth:
                self.msg_depth = None
            else:
                self.msg_rgb = None
            return False

        if self.use_multithreading:
            # we build and send the action goal in a seperate thread with a "copy" of the messages to be able to release the message lock sooner
            thread = Thread( target=self.send_action_goal, args=(self.msg_rgb, self.msg_depth, self.msg_cam_info) )
            thread.start()
            return True
        else:
            self.send_action_goal( self.msg_rgb, self.msg_depth, self.msg_cam_info )
            return True

    def send_action_goal( self, msg_rgb, msg_depth, msg_cam_info ):
        """ sends the action goal based on the provided messages """
        goal = action_msgs.semantic_segmentationGoal()
        goal.header.stamp = rospy.Time.now()
        goal.rgb = msg_rgb
        goal.depth = msg_depth
        goal.cam_info = msg_cam_info
        with self.action_lock:
            print( 'Sending Goal To Server' )
            self.action_client.send_goal( goal )
            print( 'Wait For Result From Server' )
            self.action_client.wait_for_result(timeout=self.action_timeout)
            if self.action_client.get_state() == 3: #TODO Return False on Service Call if fails 
                print( 'Got Result From Server')
                self._pub_results(self.action_client.get_result(), msg_cam_info)
            else:
                print("Server Error: Action Timeout")

    def _pub_results(self, result, msg_cam_info):
        """ Split Result Array and Publish on Topics """
        if len(result.labels) == len(result.depth):
            for label, image in result.labels, result.depth:
                publisher_image = rospy.Publisher( self.action_name + "/" + label + "/image_raw", Image, queue_size=1 )
                publisher_image.publish( image )
                publisher_info = rospy.Publisher( self.action_name + "/" + label + "/camera_info", CameraInfo, queue_size=1 )
                publisher_info.publish( msg_cam_info )
        else:
            print("Server Error: Labels And Images Do Not Match")

    def _trigger_service_cb(self, req):
        """ called when we receive a service trigger request """
        response = TriggerResponse()
        with self.msg_lock:
            result = self.check_msgs()
        response.success = result
        if result:
            response.message = "Scene Analysed"
        else: 
            response.message = "Scene Analysing Failed"
        return response

    def callback_rgb( self, msg ):
        """ called when we receive a new rgb image message """
        with self.msg_lock:
            self.msg_rgb = msg
            if not self.run_on_service:
                result = self.check_msgs()
    
    def callback_depth( self, msg ):
        """ called when we receive a new depth image message """
        with self.msg_lock:
            self.msg_depth = msg
            if not self.run_on_service:
                result = self.check_msgs()    

    def callback_cam_info( self, msg ):
        """ called when we reveive a new CameraInfo message """
        with self.msg_lock:
            self.msg_cam_info = msg
            if not self.run_on_service:
                result = self.check_msgs()

    def spin( self ):
        """ enteres the main idle loop of the node """
        rospy.spin()

if __name__ == '__main__':

    client = SceneAnalyserActionClient()
    client.spin()