#!/usr/bin/env python3

""" this code should be compatible with both python 2.7 and python 3+ """

from __future__ import print_function


from scene_analyser.octo_mapper_node import OctoMapperNode 




if __name__ == '__main__':

    print( 'starting octo-mapper node' )
    node = OctoMapperNode()
    node.spin()
    print( 'scene analyser node shut down' )


