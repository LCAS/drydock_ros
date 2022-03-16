#!/usr/bin/env python3

""" this code should be compatible with both python 2.7 and python 3+ """

from __future__ import print_function


from sensor_msgs.msg import Image
import rospy



class OctoMapperNode( object ):
    def __init__( self ):
        rospy.init_node( 'octo_mapper', anonymous=False )
        print( '{} initializing'.format(self.__class__) )
        self.subscribe()
    
    def subscribe( self ):
        print( '{} - subscribing to {}'.format(self.__class__, self.topic_rgbd) )
        rospy.Subscriber( self.topic_rgbd, Image, self.callback_rgbd )
    
    def run( self, msg ):
        pass
    
    def callback_rgbd( self, msg ):
        self.run( msg )

    def spin( self ):
        """ ros main loop """
        print( '{}: spinning'.format(self.__class__) )
        rospy.spin()





if __name__ == '__main__':

    print( 'starting octo-mapper node' )
    node = OctoMapperNode()
    node.spin()
    print( 'scene analyser node shut down' )


