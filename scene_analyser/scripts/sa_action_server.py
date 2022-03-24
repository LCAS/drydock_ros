#!/usr/bin/env python3

""" this code should be compatible with both python 2.7 and python 3+ """

from __future__ import print_function


from scene_analyser.scene_analyser_server import SceneAnalyserActionServer

#import rospy




if __name__ == '__main__':

    print( 'starting scene analyser action server' )
    act_server = SceneAnalyserActionServer()
    act_server.spin()
    print( 'scene analyser action server has shut down' )

