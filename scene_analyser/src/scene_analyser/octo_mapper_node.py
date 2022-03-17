#!/usr/bin/env python3

""" this code should be compatible with both python 2.7 and python 3+ """

from __future__ import print_function
from __future__ import division


import time
try:
    from queue import Queue # Python 3+
except:
    from Queue import Queue # Python 2.7

import numpy as np
import octomap
from octomap_msgs.msg import Octomap as OctomapMsg # renamed to avoid confusion with the octomap module above

from sensor_msgs.msg import Image, CameraInfo
import rospy



class OctoMapperNode( object ):
    
    def __init__( self ):
        rospy.init_node( 'octo_mapper', anonymous=False )
        print( '{} initializing'.format(self.__class__.__name__) )
        self.topic_image = rospy.get_param( '~topic_image', '/camera/saga_arm_d435e/aligned_depth_to_color/image_raw' ) # depth image
        self.topic_cam_info = rospy.get_param( '~topic_cam_info', '/camera/saga_arm_d435e/aligned_depth_to_color/camera_info' ) # camera info for depth image
        self.oct_resolution = 0.01
        #self.octree = octomap.OcTree( self.oct_resolution )
        self.msg_cam_info = None
        self.msg_image = Queue()
        self.subscribe()
        self.advertise()
    
    def subscribe( self ):
        """ subscribe to all ros topics that we are interested here """
        print( '{} - subscribing to {}'.format(self.__class__.__name__, self.topic_image) )
        rospy.Subscriber( self.topic_image, Image, self.callback_image )
        rospy.Subscriber( self.topic_cam_info, CameraInfo, self.callback_cam_info )
    
    def advertise( self ):
        self.publisher = rospy.Publisher( '/octomap', OctomapMsg, queue_size=1 )
        
    
    def pointcloud_from_depth( self, depth, fx, fy, cx, cy):
        """ converts a depth image into a pointcloud. based on: https://github.com/wkentaro/octomap-python/blob/main/examples/insertPointCloud.py
            fx,fy: field of view
            cx,cy: camera center
        """
        
        print( depth.shape )
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        valid = ~np.isnan(depth)
        # note: depth images with encoding type uint16 are in mm, encoding type float are in m. we need meters for our computations
        z = np.where(valid, depth*1000, np.nan)
        x = np.where(valid, z * (c - cx) / fx, np.nan)
        y = np.where(valid, z * (r - cy) / fy, np.nan)
        pc = np.dstack((x, y, z))
    
        return pc
    
    def is_ready( self ):
        if not self.msg_cam_info:
            return False # no camera info yet
        if not self.msg_image:
            return False # empty image queue
        return True
    
    def run( self ):
        if not self.is_ready():
            return
        try:
            msg = self.msg_image.get( False, 2.0 ) # non-blocking wait for up to 2 seconds. removes message from queue.
        except Exception as e:
            print( '{}.run() exception caught: {}'.format(self.__class__.__name__, e) )
            return
        time_start = time.time()
        k = self.msg_cam_info.K
        fx = k[0]
        fy = k[4]
        cx = k[2]
        cy = k[5]
        depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape( msg.height, msg.width )
        pointcloud = self.pointcloud_from_depth( depth_image, fx, fy, cx, cy )
        valid_points = ~np.isnan( pointcloud ).any( axis=2 )
        print( 'creating octomap' )
        octree = octomap.OcTree( self.oct_resolution )
        octree.insertPointCloud(
            pointcloud = pointcloud[ valid_points ],
            origin = np.array([0, 0, 0], dtype=float),
            maxrange=2 )
        print( 'finished creating octomap ({:.3f}s)'.format(time.time()-time_start) )
        msg = OctomapMsg()
        msg.data = octree
        msg.resolution = self.oct_resolution
        self.publisher.publish( msg )
        print( 'octomap published' )
    
    def callback_image( self, msg ):
        self.msg_image.put( msg )
        self.run()
        
    def callback_cam_info( self, msg ):
        """ stores the camera info. we assume that the camera info does not change over time, and simply
        overwrite it here without synchronizing with the depth images. """
        self.msg_cam_info = msg
        self.run()

    def spin( self ):
        """ ros main loop """
        print( '{}: spinning'.format(self.__class__.__name__) )
        rospy.spin()





if __name__ == '__main__':

    pass # nothign to do yet

