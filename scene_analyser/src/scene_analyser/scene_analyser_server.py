#import threading

#import rospy
#import std_msgs.msg as std_msg
import actionlib

import scene_analyser.msg as action_msgs


class SceneAnalyserActionServer( object ):
    def __init__( self, action_name="/scene_analyser" ):
        self._action_name = action_name
        self.action_server = actionlib.SimpleActionServer(
            self._action_name,
            action_msgs.semantic_segmentationAction,
            execute_cb=self.execute,
            auto_start = False)
        # todo: add any initialization that needs to be done before accepting goals
        self.action_server.start()
        print( 'Scene Analyser Action Server is ready' )

    
    def execute( self, goal ):
        """ executed when the action server receives a goal """
        print( 'scene analyser action server - goal received:' )
        print( goal )
        # todo: forward data to the scene analyser node for processing and listen to result
