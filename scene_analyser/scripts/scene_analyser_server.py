#!/usr/bin/env python3

""" this code should be compatible with both python 2.7 and python 3+ """

from __future__ import print_function

import scene_analyser.scene_analyser_node as san


from scene_analyser.scene_analyser_node import SceneAnalyserNode as Node
from scene_analyser.scene_analyser_server import SceneAnalyserActionServer

#import rospy




if __name__ == '__main__':

    print( 'starting scene analyser node' )
    print( 'foobar' )
    node = Node()
    #act_server = SceneAnalyserActionServer()
    node.spin()
    print( 'scene analyser node shut down' )

