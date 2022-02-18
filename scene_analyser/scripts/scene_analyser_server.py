#!/usr/bin/env python

""" this code should be compatible with both python 2.7 and python 3+ """

from __future__ import print_function


from scene_analyser.scene_analyser_node import SceneAnalyserNode as Node

#import rospy




if __name__ == '__main__':

    print( 'starting scene analyser node' )
    node = Node()
    node.spin()
    print( 'scene analyser node shut down' )

