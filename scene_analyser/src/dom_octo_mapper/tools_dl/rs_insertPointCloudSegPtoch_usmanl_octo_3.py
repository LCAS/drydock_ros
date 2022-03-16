#!/usr/bin/env python
from cmath import rect
from math import inf
# from black import out
from matplotlib import units
import numpy as np
from numpy.core.shape_base import block
import matplotlib.pyplot as plt
# from sensor_msgs.msg import Image, CameraInfo
#######################################################
#######################################################
from datetime import datetime
import pyrealsense2 as rs
from scipy.spatial.transform import *
import sys
########################################################
from tools_dl.rsDetect_TrackSeg_full import StawbDetTracker
# from tools.rs_insertPointCloudepth_octo import rs_callback_xyz_octo
from MaskPredictor.usman_dl import call_predictor

import os, time,math
import cv2, json
from collections import OrderedDict
######################################

import glooey
from glooey.containers import VBox
import pyglet
from pyglet import shapes
import trimesh
import trimesh.transformations as tf
import trimesh.viewer
import octomap
######################################
from itertools import count

# https://scipy-lectures.org/intro/scipy/auto_examples/plot_connect_measurements.html
def pointcloud_from_depth(depth, fx, fy, cx, cy):
    if type(depth) is not None:
        # assert depth.dtype.kind == 'f', 'depth must be float and have meter values'
        assert depth.dtype == 'f', 'depth must be float and have meter values'

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = ~np.isnan(depth)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - cx) / fx, np.nan)
    y = np.where(valid, z * (r - cy) / fy, np.nan)
    pc = np.dstack((x, y, z))

    return pc

    ##############################################
    ##############################################


class rastrack():
    """rasberry detection class

    each detection frame is composed of a ROI, an ID and a Kalman filter
    so we create a rasberry class to hold the object state
    """
    _ids = count(1)  
    # link Kalman with Object（self.kalman）
    def __init__(self, id, track_pt,track_window,strawb,depth_filtered,color_filtered,occupied, empty):
        """init the pedestrian object with track window coordinates"""
        # set up the roi
        self.id = id
        #counting instances of a grasstrack
        self.numofids  = next(self._ids)
        print('total of ids = {0}, lane ID = {1}'.format(self.numofids,self.id))
        self.track_win = track_window
        self.center = track_pt
        self.roi = strawb #frame[y-h:y, x-w:x]#cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)        
        self.depth_obj = depth_filtered  # keep individual segmented obj depth image
        self.color_obj = color_filtered  # hold individual segmented obj color image
        self.occupied = occupied
        self.empty = empty
        self.depth_points = []
        self.vertices_interest = None
        self.color_interest = None

    def __del__(self):
        print ("lane %d destroyed" %self.id)

class MapManager(object):
    def __init__(self):

        self.aligned_frames = None  # choose webcam False: choose realsense camera
        self.depth_frame = None
        self.color_frame = None
        self.depth_image = None
        self.color_image = None
        self.depth_octo = None
        self.color_octo = None
        self.oct_resolut = 0.01 # 0.01

        self.rgbd_image = None  #  place holder for usman's output of original color [:,:,:3] and depth [:,:, 3]
        self.depth_masks = None   #  place holder for usman's output of segmentation

        self.num = -1
        self.num_images = inf
        self.figsave_path = 'results/'
        # self.depth_image_oct   = 0.0
        self.messages_img=None
        self.messages_alignedepth = None

        self.points = None # pc.calculate(mapObj.depth_frame)  
        self.vertices = None #np.asanyarray(mapObj.points.get_vertices(dims=2))
        self.image_Points = None #np.reshape(vertices , (-1,width,3))
        self.vertices_interest = None #mapObj.image_Points[y:h, x:w,:].reshape(-1,3)
        self.color_interest = None #color_image[y:h, x:w,:].reshape(-1,3)
        ###############################
        ##### realtime camera settings ---- initialization

        self.pipeline = rs.pipeline()  # Define the process pipeline        
        self.config = rs.config()   # Define configuration config
        self.depth_scale = 0.001   # default value

        self.bagfileFlg =True
        if self.bagfileFlg ==False:
            # enable from live camera
            self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # To configure depth flow 
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)   # To configure color flow 
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()
        
        elif self.bagfileFlg ==True:
            # enable from .bag files
            self.config.enable_device_from_file("dataset/20210827_121930.bag")
        
        self.profile = self.pipeline.start(self.config)  # The process begins 
        self.align_to = rs.stream.color  # And color Flow alignment 
        self.align = rs.align(self.align_to)

        
        self.clipping_distance_in_meters = 1.0 # 40cm以内を検出
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

        ####### pcd ##################
        
        self.pc = rs.pointcloud()
        # We want the points object to be persistent so we can display the last cloud when a frame drops
        self.points = rs.points()
        # direct procssing 3d from realsense camera
        self.colorizer = rs.colorizer()

        ###### real time setting above
        #################################

        self.grasslanes = {} # hold detection objets
        
        self.obj_data = None
        self.ctl_data = None
        # self.camera_info = CameraInfo()
        self.width = 1280
        self.height = 720
        self.fx = 908.594970703125
        self.fy = 908.0234375
        self.ppx = 650.1152954101562
        self.ppy = 361.9811096191406
        self.output_path = 'results'
        if os.path.isdir(self.output_path)==False:
            os.mkdir(self.output_path)
            print("self.output_path '% s' created" % self.output_path)


        self.flag = True

        self.K = np.array([[self.fx, 0, self.ppx],
                           [0, self.fy, self.ppy],
                           [0, 0, 1]])

        self.K[0, 0] = self.fx
        self.K[1, 1] = self.fy
        self.K[0, 2] = self.ppx
        self.K[1, 2] = self.ppy

        self.depth_intrin = rs.pyrealsense2.intrinsics()
        self.color_intrin = None #rs.pyrealsense2.intrinsics()
        self.depth_to_color_extrin = None
        self.depth_sensor = None
        self.depth_scale = 0.001
        # Intrinsics & Extrinsics
        self.fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
        self.fourcc2 = cv2.VideoWriter_fourcc('M','J','P','G')
        self.avi_width = 2560
        self.avi_height = 720
        self.outRes = None
        # self.depth_intrin =  [ 1280x720  p[650.115 361.981]  f[908.595 908.023]  Inverse Brown Conrady [0 0 0 0 0] ]
        self.depth_intrin.width =1280
        self.depth_intrin.height=720
        self.depth_intrin.ppx=650.1152954101562
        self.depth_intrin.ppy=361.9811096191406
        self.depth_intrin.fx = 908.594970703125
        self.depth_intrin.fy = 908.0234375
        self.depth_intrin.model=rs.distortion.none
        self.depth_intrin.coeffs=[0.0, 0.0, 0.0, 0.0, 0.0]
        # open3d processing
        # display with o3d- Non blocking
        
        # self.vis = o3d.visualization.Visualizer ()
        # self.vis.create_window ( 'PCD' , width = 1280 , height = 720,visible = False)

        # pointcloud = o3d.geometry.PointCloud ()
        # self.geom_added = False
        # intrinsics = self.depth_intrin  #frm_profile.as_video_stream_profile().get_intrinsics ()
        # self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic (self.depth_intrin.width, self.depth_intrin.height, self.depth_intrin.fx, self.depth_intrin.fy, self.depth_intrin.ppx, self.depth_intrin.ppy)
   
        
        print('simu_depth_intrin = ',self.depth_intrin)

        self.lowerLimit = np.array([150, 150, 60], np.uint8)
        self.upperLimit = np.array([179, 255, 255], np.uint8)
        #####################################################
        self.rsDetTrck = StawbDetTracker()
        # detection and tracking points from segmentation in image plate: (x,y)
        self.centers = None
        self.depth_point = None  # 3D position [x, y, z]
        #####################################################
        # for purpose of open3D usage
        # Set camera params.
        # self.camera_parameters = o3d.camera.PinholeCameraParameters()
        # self.camera_parameters.intrinsic.set_intrinsics(width=self.depth_intrin.width, height=self.depth_intrin.height, fx=self.K[0][0], fy=self.K[1][1], cx=self.K[0][2], cy=self.K[1][2])
        # # self.camera_parameters.extrinsic = np.array([[1.204026265313826, 0, 0, -0.3759973645485034],
        # #                                    [0, -0.397051999192357, 0, 4.813624436153903, ],
        # #                                    [0, 0, 0.5367143925232766, 7.872266818189111],
        # #                                    [0, 0, 0, 1]])
        # self.camera_parameters.extrinsic = np.array([[1, 0, 0, 0],
        #                                    [0, 1, 0, 0],
        #                                    [0, 0, 1, 0],
        #                                    [0, 0, 0, 1]])
    
    # https://codereview.stackexchange.com/questions/79032/generating-a-3d-point-cloud
    def point_cloud(self, depth):
        """Transform a depth image into a point cloud with one point for each
        pixel in the image, using the camera transform for a camera
        centred at cx, cy with field of view fx, fy.

        depth is a 2-D ndarray with shape (rows, cols) containing
        depths from 1 to 254 inclusive. The result is a 3-D array with
        shape (rows, cols, 3). Pixels with invalid depth in the input have
        NaN for the z-coordinate in the result.
        self.width = 1280
        self.height = 720
        self.fx = 908.594970703125
        self.fy = 908.0234375
        self.ppx = 650.1152954101562
        self.ppy = 361.9811096191406

        """
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        valid = (depth > 0) & (depth < 255)
        z = np.where(valid, depth / 256.0, np.nan)
        x = np.where(valid, z * (c - self.ppx) / self.fx, 0)
        y = np.where(valid, z * (r - self.ppy) / self.fy, 0)

        return np.dstack((x, y, z))

    
    def get_aligned_images(self):

        self.frames = self.pipeline.wait_for_frames()  # Wait for image frame 
        aligned_frames = self.align.process(self.frames)  # Get alignment frame 
        self.depth_frame = aligned_frames.get_depth_frame()  # Gets the in the aligned frame depth frame 
        self.color_frame = aligned_frames.get_color_frame()   # Gets the in the aligned frame color frame 

        ###############  Acquisition of camera parameters  #######################
        self.depth_to_color_extrin = self.depth_frame.profile.get_extrinsics_to(self.color_frame.profile)

        if self.num <10:
            self.color_intrin = self.color_frame.profile.as_video_stream_profile().intrinsics   # Get camera internal parameters 
            self.depth_intrin = self.depth_frame.profile.as_video_stream_profile().intrinsics  # Get the depth parameter （ Pixel coordinate system to camera coordinate system will use ）
            self.depth_to_color_extrin = self.depth_frame.profile.get_extrinsics_to(self.color_frame.profile)

            camera_parameters = {
            'fx': self.color_intrin.fx, 'fy': self.color_intrin.fy,
                                'ppx': self.color_intrin.ppx, 'ppy': self.color_intrin.ppy,
                                'height': self.color_intrin.height, 'width': self.color_intrin.width,
                                'depth_scale': self.profile.get_device().first_depth_sensor().get_depth_scale()
                                }
            #  Save internal reference to local 
            param_pth = os.path.join(self.output_path,'intr7insics.json')
            with open(param_pth, 'w') as fp:
                json.dump(camera_parameters, fp)
            #######################################################
        
        self.depth_image = np.asanyarray(self.depth_frame.get_data())  # Depth map （ Default 16 position ）
        self.depth_image_8bit = cv2.convertScaleAbs(self.depth_image, alpha=0.03)  # Depth map （8 position ）
        self.depth_image_3d = np.dstack((self.depth_image_8bit,self.depth_image_8bit,self.depth_image_8bit))  #3 Channel depth map 
        self.color_image = np.asanyarray(self.color_frame.get_data())  # RGB chart 
        
        #######################
        self.colorized_depth = np.asanyarray(self.colorizer.colorize(self.depth_frame).get_data())
        # plt.imshow(colorized_depth)
         # clipping_distance_in_metersm以内を画像化
        self.white_color = 255 # 背景色
        self.bg_removed = np.where((self.depth_image_3d > self.clipping_distance) | (self.depth_image_3d <= 0), self.white_color, self.color_image)
        # 背景色となっているピクセル数をカウント
        self.white_pic = np.sum(self.bg_removed == 255)
        #######################
        
        #######################

        # Return camera internal parameters 、 Depth parameter 、 Color picture 、 Depth map 、 In homogeneous frames depth frame 
        return self.color_intrin, self.depth_intrin, self.color_image, self.depth_image, self.depth_frame
    '''  Obtain the 3D coordinates of random points  '''
    def get_3d_camera_coordinate(self,depth_pixel, aligned_depth_frame, depth_intrin):
        x = depth_pixel[0]
        y = depth_pixel[1]
        dis = aligned_depth_frame.get_distance(x, y)        #  Get the depth corresponding to the pixel 
        # print ('depth: ',dis) #  The unit of depth is m
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
        # print ('camera_coordinate: ',camera_coordinate)
        return dis, camera_coordinate     
 
          
        
         
    def Pos2DPixels3Dxyz(self,depth_frame,c, r, depth_intrin):
        depth = depth_frame.get_distance(c, r)
        #THE FOLLOWING CALL EXECUTES FINE IF YOU PLUGIN A BOGUS DEPH
        depth_point_mts_cam_coords = rs.rs2_deproject_pixel_to_point(depth_intrin, [c, r], depth)
        return depth ,depth_point_mts_cam_coords
    def labeled_scene_widget(self,scene, label):
        vbox = glooey.VBox()
        vbox.add(glooey.Label(text=label, color=(255, 255, 255)), size=0)
        vbox.add(trimesh.viewer.SceneWidget(scene))
        # fig = plt.figure('trimesh.viewer', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
        # plt.title('trimesh.viewer')
        # plt.imshow(trimesh.viewer.SceneWidget(scene),cmap='gray')
        # plt.show(block=True)
        # fig = trimesh.viewer.SceneWidget(scene)    
        return vbox

    def visualize_xyz(self,centers,depth_frame,depth_intrin,occupied, empty, K, width, height, rgb, pcd, mask, aabb):

        # sample_buffers to get better looking drawings 

        # You can use this to support newer hardware features where available, but also accept a lesser config if necessary. For example, the following code creates a window with multisampling if possible, otherwise leaves multisampling off:
        # https://pyglet.readthedocs.io/en/latest/programming_guide/context.html
        # config = pyglet.gl.Config(sample_buffers=1, samples=1, double_buffer=False)
        """
        self.visualize_xyz(centers,self.depth_frame,self.depth_intrin,
            occupied=occupied,
            empty=empty,
            K=self.K,
            width=self.depth_intrin.width,
            height=self.depth_intrin.height,
            rgb=self.color_octo,
            pcd=pcd,
            mask=mask,
            aabb=aabb#(aabb_min, aabb_max),)
        
        
        context = config.create_context(None)
        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        
        template = pyglet.gl.Config(sample_buffers=1, samples=1)
        try:
            config = screen.get_best_config(template)
        except pyglet.window.NoSuchConfigException:
            template = pyglet.gl.Config()
            config = screen.get_best_config(template)"""


        # window = pyglet.window.Window(config=config)
        # template = pyglet.gl.Config(alpha_size=8)
        # config = screen.get_best_config(template)
        # context = config.create_context(None)
        # window = pyglet.window.Window(context=context)
        config = pyglet.gl.Config(depth_size = 24,sample_buffers=1, samples=8, double_buffer=False)  
        # window = pyglet.window.Window(width=int(width * 0.5 * 3), height=int(height * 0.75),config = config,resizable=True)
        window = pyglet.window.Window(width=int(width * 0.5 * 3), height=int(height * 0.75),config = config,resizable=True)
        ##############################################################
        # testing on text window drawing - dom
        recxy = 10
        for count, value in enumerate(centers):
            print(count, value)
            cx = int(centers[count][0])
            cy = int(centers[count][1])
            if cx<recxy or cy<recxy:
                break
            #################################################################################
            # 3D translation and transformation with centroilds, realsense. open3d
            #################################################################################
            # Detect object maxctroid: the largest contour
            distance, depth_point = self.Pos2DPixels3Dxyz(depth_frame,cx,cy,depth_intrin)
            text = "xyz: %.5lf, %.5lf, %.5lfm" % (depth_point[0], depth_point[1], depth_point[2])
            # cv2.putText(color_frame, "Measured xyz{}m".format(distance), (depth_point[0], depth_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            # cv2.putText(frame, text, (int(depth_point[0]), int(depth_point[1]) - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            # cv2.putText(frame,text, (cx + recxy, cy - recxy), 0, 0.5, (0,200,255), 2)
            # cv2.circle(frame, (cx,cy), 8, (0, 0, 255), 1)
            # drawn on colormap - acting as mesh at momoment:
            
            # cv2.putText(depth_image,text, (cx + recxy, cy - recxy), 0, 0.5, (0,200,255), 2)
            # cv2.circle(depth_image, (cx,cy), 8, (0, 0, 255), 1)
            # creating alabel
            scale_w =2#0.5# two crossed? 2 # second
            label = pyglet.text.Label('strawb',
                                    font_name ='Times New Roman',
                                    color=(255,0, 0, 255),
                                    font_size = 36,
                                    x = scale_w*cx,#scale_w*window.width//2, 
                                    y = scale_w*cy,#window.height//2,
                                    anchor_x ='center', anchor_y ='center')

            """        
            ##############################################################
            # creating alabel
            scale_w =0.5  # 1 - one image to display, 0.5 - two image
            label = pyglet.text.Label('strawb',
                                    font_name ='Times New Roman',
                                    color=(255,0, 0, 255),
                                    font_size = 36,
                                    x = scale_w*window.width//2, y = window.height//2,
                                    anchor_x ='center', anchor_y ='center')"""

        # window = pyglet.window.Window(width=int(width * 0.25 * 3), height=int(height * 0.75),config=config,resizable=True)
        # batch = pyglet.graphics.Batch()
        # line = shapes.Line(100, 100, 50, 200,color=(255,0,0), width=19, batch=batch)
        @window.event
        def on_key_press(symbol, modifiers):
            if modifiers == 0:
                if symbol == pyglet.window.key.Q:
                    window.on_close()

        gui = glooey.Gui(window)
        hbox = glooey.HBox()
        hbox.set_padding(5)

        camera = trimesh.scene.Camera(
            resolution=(width, height), focal=(K[0, 0], K[1, 1])
        )
        camera_marker = trimesh.creation.camera_marker(camera, marker_height=0.1)

        # initial camera pose
        # camera_transform = np.array(
        #     [
        #         [0.73256052, -0.28776419, 0.6168848, 0.66972396],
        #         [-0.26470017, -0.95534823, -0.13131483, -0.12390466],
        #         [0.62712751, -0.06709345, -0.77602162, -0.28781298],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ],
        # )

        camera_transform = np.array(
            [
                [1.0, -0.0, 0.0, 0.0],
                [-0.0, -1, -0.0, -0.0],
                [0.0, -0.0, -1.0, -0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )

        aabb_min, aabb_max = aabb
        bbox = trimesh.path.creation.box_outline(
            aabb_max - aabb_min,
            tf.translation_matrix((aabb_min + aabb_max) / 2),
        )

        geom = trimesh.PointCloud(vertices=pcd[mask], colors=rgb[mask])
        scene = trimesh.Scene(camera=camera, geometry=[bbox, geom, camera_marker])
        scene.camera_transform = camera_transform
        VBox= self.labeled_scene_widget(scene, label='pointcloud')
        hbox.add(VBox)  


        
        # color_scene = np.asanyarray(VBox,dtype=np.uint8)       
        # num = 100
        # figname='strw_octomap_depth_rgb_'+str(num)+'.png'   
        # nampepath = os.path.join(output_path, figname) 
        # # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(nampepath,color_scene)
        # fig = plt.figure('scene', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
        # plt.title('scene')
        # plt.imshow(hbox,cmap='gray')
        # plt.show(block=True)



        geom = trimesh.voxel.ops.multibox(
            occupied, pitch=self.oct_resolut, colors=[1.0, 0, 0, 0.5]
        )
        scene = trimesh.Scene(camera=camera, geometry=[bbox, geom, camera_marker])
        scene.camera_transform = camera_transform
        hbox.add(self.labeled_scene_widget(scene, label='occupied'))
        # viewplet = pyglet.app.run()
        # return

        geom = trimesh.voxel.ops.multibox(
            empty, pitch=self.oct_resolut, colors=[0.5, 0.5, 0.5, 0.5]
        )
        scene = trimesh.Scene(camera=camera, geometry=[bbox, geom, camera_marker])
        scene.camera_transform = camera_transform
        hbox.add(self.labeled_scene_widget(scene, label='empty'))

        gui.add(hbox)  
        # @window.event
        # def on_draw():
        # #     # window.clear()
        # #     batch.draw()
        #     pyglet.gl.glFlush()  
    
        # pyglet.app.EventLoop
        # sprite = pyglet.sprite.Sprite(img=hbox)
        # @window.event
        # def on_draw():
        #     window.clear()
        #     sprite.draw()
        # display = pyglet.canvas.get_display() # for imaage only, due to need im.texture
        event_loop = pyglet.app.EventLoop()
        @event_loop.event
        def on_window_close(window):
            event_loop.exit()
            return pyglet.event.EVENT_HANDLED
        # viewplet = pyglet.app.run()
        # making window hide
        # setting visible property of the window
        # window.set_visible(False)
        # event_loop.run()
        # close the window
        # window.close()
        # https://www.geeksforgeeks.org/pyglet-window-deactivate-event/
        @window.event   
        # window deactivate event      This event is triggered when we switched to another app
        def on_deactivate():      
            # printing message
            print("Switched to another app")
        viewplet = pyglet.app.run()
        return
        # 
        # fig = plt.figure('scene', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
        # plt.title('scene')
        # plt.imshow(viewplet,cmap='gray')
        # plt.show(block=True)

    def pointcloud_from_depth(self,depth, fx, fy, cx, cy):
        assert depth.dtype.kind == 'f', 'depth must be float and have meter values'

        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        valid = ~np.isnan(depth)
        z = np.where(valid, depth, np.nan)
        x = np.where(valid, z * (c - cx) / fx, np.nan)
        y = np.where(valid, z * (r - cy) / fy, np.nan)
        pc = np.dstack((x, y, z))

        return pc
    # def rs_callback_xyz_octo(self,pcd, occupied,empty, mask,resolution, aabb, centers):#self,depth_image,rgb,K,depth_intrin,depth_frame,num,output_path,centers=[],debugMode=True):       
    def rs_callback_xyz_octo(self,depth_image,rgb,K,depth_intrin,depth_frame,num,output_path,centers=[],debugMode=True):

        # data = imgviz.data.arc2017()
        """    
        camera_info = data['camera_info']
        K = np.array(camera_info['K']).reshape(3, 3)
        rgb = data['rgb']
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_image = data['depth']"""
        # lowerLimit = np.array([150, 150, 60], np.uint8)
        # upperLimit = np.array([179, 255, 255], np.uint8)
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # # Stack both images horizontally
        # src2 = np.zeros_like(rgb)
        # src2[:,:,0] = depth_image
        # src2[:,:,1] = depth_image
        # src2[:,:,2] = depth_image
        # images = np.hstack((src2,rgb))
        # fig = plt.figure('rs_callback', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
        # plt.title('Rs Depth & Colors inside octomap')
        # plt.imshow(images,cmap='gray')
        # plt.show(block=False)
        # plt.pause(0.5)
        # plt.close()

        pcd = self.pointcloud_from_depth(
            depth_image, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )


        outs_pcd = {}

        outs_pcd['X'] = pcd[:, 0]
        outs_pcd['Y'] = pcd[:, 1]
        outs_pcd['Z'] = pcd[:, 2]
        print("outs_pcd['X']", outs_pcd['X'])

        xyzx = np.reshape(outs_pcd['X'], -1)
        xyzy = np.reshape(outs_pcd['Y'], -1)
        xyzz = np.reshape(outs_pcd['Z'], -1)
        print(" xyzx",  xyzx)

        # length of array
        # n = pcd.size
        # pcd = pcd.reshape(720,1280,3)
        nonnan = ~np.isnan(pcd).any(axis=2)
        mask = np.less(pcd[:, :, 2], 2)

        # self.oct_resolut = 0.01 # 0.01
        octree = octomap.OcTree(self.oct_resolut)
        octree.insertPointCloud(
            pointcloud=pcd[nonnan],
            origin=np.array([0, 0, 0], dtype=float),
            maxrange=2,
        )
        occupied, empty = octree.extractPointCloud()
        # fig = plt.figure('octree', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
        # plt.title('occupied & empty')
        # plt.imshow(octree,cmap='gray')
        # plt.show(block=True)        

        aabb_min = octree.getMetricMin()
        aabb_max = octree.getMetricMax()
        
        """
        ##########get detection and measurement done!##########
        # Segmentation with Octmap : from PCD to Octomap
        # color_image_aligned,depth_colormap_aligned = pcd2Octomap(depth_image,color_image)
        #################################################################################
        # Detect object maxctroid: the largest contour 
        recxy = 20
        centers, frame = detectMore_trackOne(rgb,lowerLimit,upperLimit,num,debugMode)
        for count, value in enumerate(centers):
            print(count, value)
            cx = int(centers[count][0])
            cy = int(centers[count][1])
            if cx<recxy or cy<recxy:
                break
            #################################################################################
            # 3D translation and transformation with centroilds, realsense. open3d
            #################################################################################
            # Detect object maxctroid: the largest contour
            # the xyz calculated using realsense camera built in function
            distance, depth_point = Pos2DPixels3Dxyz(depth_frame,cx,cy,depth_intrin)
            text = "xyz: %.5lf, %.5lf, %.5lfm" % (depth_point[0], depth_point[1], depth_point[2])
            # cv2.putText(color_frame, "Measured xyz{}m".format(distance), (depth_point[0], depth_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            # cv2.putText(frame, text, (int(depth_point[0]), int(depth_point[1]) - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            cv2.putText(frame,text, (cx + recxy, cy - recxy), 0, 0.5, (0,200,255), 2)
            cv2.circle(frame, (cx,cy), 8, (0, 0, 255), 1)
            # drawn on colormap - acting as mesh at momoment:
            
            cv2.putText(depth_image,text, (cx + recxy, cy - recxy), 0, 0.5, (0,200,255), 2)
            cv2.circle(depth_image, (cx,cy), 8, (0, 0, 255), 1)

            figname='strw_xyz_msh'+str(num)+'.png'
            # os.chdir(OUTDIR)
            nampepath = os.path.join(output_path, figname) 
            # depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
            cv2.imwrite(nampepath,frame)
        #######################################################
        """    
        """
        self.visualize_xyz(centers,self.depth_frame,self.depth_intrin,
            occupied=occupied,
            empty=empty,
            K=self.K,
            width=self.depth_intrin.width,
            height=self.depth_intrin.height,
            rgb=self.color_octo,
            pcd=pcd,
            mask=mask,
            aabb=aabb#(aabb_min, aabb_max),
        )
        """
        self.visualize_xyz(centers,depth_frame,depth_intrin,
        occupied=occupied,
        empty=empty,
        K=K,
        width=depth_intrin.width,
        height=depth_intrin.height,
        rgb=rgb,
        pcd=pcd,
        mask=mask,        
        aabb=(aabb_min, aabb_max),
    )

        # visualize_xyz(centers,depth_frame,depth_intrin,
        #     occupied=occupied,
        #     empty=empty,
        #     K=K,
        #     width=camera_info['width'],
        #     height=camera_info['height'],
        #     rgb=rgb,
        #     pcd=pcd,
        #     mask=mask,
        #     aabb=(aabb_min, aabb_max),
        # )
        return occupied, empty 

    def rs_callback_cvSeg(self,depth_image,rgb_image):

        self.rgbd_image,self.depth_masks = call_predictor(rgb_image,depth_image)


    def rs_callback_octomapping(self,img_inst,img_ori,minarea = 100, maxarea = 400):

        #1.
        listimages = [self.depth_image]
        listtitles = ["depth_image"] 
        #.2
        listimages.append(img_ori)
        listtitles.append("rgb_image")            

        height, width = img_inst.shape
        # img = np.zeros([height, width, 1], dtype=np.uint8)
        img = np.zeros([height, width], dtype=np.uint8)
        # info = np.iinfo(img_inst.dtype) # Get the information of the incoming image type
        # data = img_inst.astype(np.float64) / info.max # normalize the data to 0 - 1
        # data = 255 * data # Now scale by 255
        # img = data.astype(np.uint8)
        
        # _img_inst = np.copy(img_inst)
        # img[img_inst>0.1] = 0.0
        img_inst_ch = img_inst#[:,:,0]
        img[img_inst_ch>0.0] = 255   
        #3.
        listimages.append(img)
        listtitles.append("img_inst")
        # gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # converting to its binary form
        # ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
        kernel = np.ones((15, 15), np.uint8)
        self.inst_image = cv2.dilate(img, kernel, iterations=1)

        # kernel = np.ones((3, 3), np.uint8)
        # self.inst_image = cv2.erode(self.inst_image, kernel, iterations=1)
        # kernel = np.ones((5, 5), np.uint8)
        # self.inst_image = cv2.erode(self.inst_image, kernel, iterations=1)
        # kernel = np.ones((17, 17), np.uint8)
        # self.inst_image = cv2.dilate(self.inst_image, kernel, iterations=1)
        # Opening is erosion operation followed by dilation operation.       
        # self.inst_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # self.inst_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # numpy_vertical = np.vstack((image, grey_3_channel))
        # numpy_horizontal = np.hstack((img, self.inst_image))
        # numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
        # numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
        """        
        cv2.imshow('Ori-Dilated Image', numpy_horizontal)
        cv2.waitKey(500)
        cv2.destroyAllWindows()    
        """   


        recxy = 10     
        h, w, ch = self.color_image.shape
        dim = (w,h)
        self.inst_image = cv2.resize(self.inst_image,dim, interpolation=cv2.INTER_LINEAR) 

        #4.
        listimages.append(self.inst_image)
        listtitles.append("inst_image enhanced")           
        
        #find all connected components (blobs in image)
        output = np.zeros((self.inst_image.shape[0], self.inst_image.shape[1], 3), np.uint8) 

        # depth_copy = self.depth_image[:,:,0].copy()  # for usman's data
        depth_copy = self.depth_image.copy()
        inst_copy = self.inst_image.copy()#img_inst_ch.copy()

        inst_mask = inst_copy>0#img_inst_ch>0
        inst_mask = np.array(inst_mask, dtype=np.ubyte)

        inst_copy[inst_mask] = 1
        inst_copy[~inst_mask] = 0
        inst_copy = np.array(inst_copy, dtype=np.ubyte)
        """
        ### verify depth for octomap##################
        self.depth_octo = cv2.bitwise_and(self.depth_octo, self.depth_octo, mask=inst_mask) 
        self.color_octo = cv2.bitwise_and(self.color_octo, self.color_octo, mask=inst_mask)     
        occupied, empty  = rs_callback_xyz_octo(self.depth_octo,self.color_octo,self.K,self.depth_intrin,self.depth_frame,self.num,self.figsave_path)
        ##############################################
        """  
    
        ################################################################
        ####### Dom's cripts - two methods for octree: pcd and voxel
        ################################################################
        # points = pcd.points
        # Pointcloud data to arrays for pc from realsense camera
        # v, t = points.get_vertices(), points.get_texture_coordinates()
        # verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        # texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
           
        #######################
        figsize =(16, 12) # [12,8]
        name_pref = 'cropping_pcd'
        # create figure (fig), and array of axes (ax)
        numofresuts = len(listimages)
        nrows = int(math.sqrt(numofresuts))
        ncols = round(numofresuts/nrows)
        print('numofresuts = ',len(listtitles))
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i in range(1, ncols*nrows +1):#i in range(1, ncols*nrows +1):   # for i in range(1,ncols*(nrows-1)+1):
                # img = np.random.randint(10, size=(h,w))
                fig.add_subplot(nrows, ncols, i)
                # plt.subplot(nrows,ncols,i+ncols)
                plt.imshow(listimages[i-1],'gray') #alpha=0.25)
                plt.title(listtitles[i-1])
                # ax[-1].set_title("ax:"+str(i-1)+listtitles[i-1])  # set title
                plt.xticks([]),plt.yticks([])

        plt.gcf().canvas.set_window_title(name_pref)   

        name = 'pcd_obj_inhance_sum_'+str(self.num)+'.png' 
        sved_inst = self.figsave_path+name   
        plt.savefig(sved_inst, bbox_inches='tight')     
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()  

        # out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')

        # Identify the x and y positions of all nonzero pixels in the image 
        # - indices is same as dimention of row and col , which is clever part applied bebow
        # nonzero = self.depth_filtered.nonzero()
        # nonzeroy = np.array(nonzero[0])
        # nonzerox = np.array(nonzero[1])

        # Python program explaining nonzero() function 
        # arr = np.array([[0, 8, 0], [7, 0, 0], [-5, 0, 1]]) 
        # print ("Input array : \n", arr) 
        
        # out_tpl = np.nonzero(self.depth_filtered>0) 
        # print ("Indices of non zero elements : ", out_tpl) 

        # nonzeroout_tply = np.array(out_tpl[0])
        # nonzeroout_tplx = np.array(out_tpl[1])
        # print ("Output array of non-zero number y: ", nonzeroout_tply) 
        # print ("Output array of non-zero number x: ", nonzeroout_tplx) 

        # # Python program for getting The corresponding non-zero values: 
        # out_arr = self.depth_filtered[np.nonzero(self.depth_filtered>0)] 
        # print ("Output array of non-zero number: ", out_arr) 


        # # Python program for grouping the indices by element, rather than dimension     
        # out_ind = np.transpose(np.nonzero(self.depth_filtered))
        # print ("indices of non-zero number: \n", out_ind) 
        
        # a = np.array(list(zip(nonzeroy,nonzerox)))
        # b = zip(nonzeroy,nonzerox)
        # c = np.stack((nonzeroy,nonzerox), axis = 1)
        # d = np.column_stack((nonzeroy,nonzerox))
        # out_ind_zip = a
        # print ("indices of non-zero number out_ind_zip: \n", out_ind_zip)        
        #######################
        strawbs_dic = {}
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.inst_image, connectivity=8)
        # handle for octomap:
        # mapObj.depth_octo = 0.001*np.asanyarray(mapObj.depth_frame.get_data(),dtype=np.float32)

        # left_lane_inds = []
        # right_lane_inds = []
        # minarea = 100
        # #1. initialize container for debuging
        # maxarea = 400
        for lkeys in range(1, num_labels,1):
            
            listimages=[]
            listtitles=[]
            x, y, w, h, area = stats[lkeys]
            print('area = :', area)
            if area <maxarea:
                continue
                        
            cx, cy = centroids[lkeys]
            print('x, y, w, h, s, cx,cy =:',x, y, w, h,cx,cy)
            left_top = (x, y)
            right_bottom = (x + w, y + h)
            self.color_image = cv2.rectangle(self.color_image, left_top, right_bottom, (0, 0, 255), 4)
            self.color_image = cv2.putText(self.color_image, str(area), left_top, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            self.color_image = cv2.circle(self.color_image, (int(cx), int(cy)), 4, (0, 255, 0))

            listimages.append(self.color_image)
            listtitles.append('color_image with detected objs')

            output = np.zeros((self.inst_image.shape[0], self.inst_image.shape[1], 3), np.uint8)          

            ###############################################
            roi = self.inst_image[y:y+h, x:x+w]
            # cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT
            lblareas = stats[1:,cv2.CC_STAT_AREA]
            imax = max(enumerate(lblareas), key=(lambda x: x[1]))[0] + 1
            # boundingrect =  cv2.rectangle(stats[imax, cv2.CC_STAT_LEFT],
            #                     stats[imax, cv2.CC_STAT_TOP], 
            #                     stats[imax, cv2.CC_STAT_WIDTH], 
            #                     stats[imax, cv2.CC_STAT_HEIGHT])
            # boundingrect.addoffset((x,y))
          

            ###############################################
            depth_image_obj = self.depth_image.copy()
            depth_copy = self.depth_image.copy()
            mask = labels == lkeys
            output[:, :, 0][mask] = np.random.randint(0, 255)
            output[:, :, 1][mask] = np.random.randint(0, 255)
            output[:, :, 2][mask] = np.random.randint(0, 255)   

            listimages.append(output)
            listtitles.append("output overall masked with 1")

            # enhance mask
            kernel = np.ones((21, 21), np.uint8)
            output[:, :, 0] = cv2.dilate(output[:, :, 0], kernel, iterations=1)
            ###############################################
            ##########singl obj cotomap####################
            depth_octo_obj = cv2.bitwise_and(self.depth_octo, self.depth_octo, mask=output[:, :, 0])
            color_octo_obj = cv2.bitwise_and(self.color_octo, self.color_octo, mask=output[:, :, 0])  


            occupied, empty  = self.rs_callback_xyz_octo(depth_octo_obj,color_octo_obj,self.K,self.depth_intrin,self.depth_frame,self.num,self.figsave_path)
            ################################################
            ###############################################
            depth_filtered = cv2.bitwise_and(depth_copy, depth_copy, mask=output[:, :, 0])
            self.depth_filtered = depth_filtered.astype(np.uint16)
            listimages.append(self.depth_filtered)
            listtitles.append('bitwise_and by enhanced 1 obj mask')
            
            depth_image_obj[~mask] = 0.0
            listimages.append(depth_image_obj)
            listtitles.append('depth image filtered by single obj')
            #######################################################################
            # try another way to obtain the individual obj corrresponding xyz points
            """
            Transform a depth image into a point cloud with one point for each
            pixel in the image, using the camera transform for a camera
            centred at cx, cy with field of view fx, fy.

            depth is a 2-D ndarray with shape (rows, cols) containing
            depths from 1 to 254 inclusive. The result is a 3-D array with
            shape (rows, cols, 3). Pixels with invalid depth in the input have
            NaN for the z-coordinate in the result.
            
            rows, cols = depth_image_obj.shape
            c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
            valid = (depth_image_obj > 0) & (depth_image_obj < 255)
            z = np.where(valid, depth_image_obj / 256.0, np.nan)
            x = np.where(valid, z * (c - self.cx) / self.fx, 0)
            y = np.where(valid, z * (r - self.cy) / self.fy, 0)
            return np.dstack((x, y, z))
            """
            # npts_xyz = self.point_cloud(depth_image_obj)
            # print('npts_xyz [0]: ', npts_xyz[240])
            ########################################################################            
            
            ############get sigle object pcd using open3D ##################
            color_copy = img_ori.copy()
            self.color_filtered = cv2.bitwise_and(color_copy, color_copy, mask=output[:, :, 0])  
            #################################################
            # # create figure (fig), and array of axes (ax)
            # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            numofresuts = len(listimages)
            nrows = int(math.sqrt(numofresuts))
            ncols = round(numofresuts/nrows)
            print('numofresuts = ',len(listtitles))
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            for i in range(1, ncols*nrows +1):#i in range(1, ncols*nrows +1):   # for i in range(1,ncols*(nrows-1)+1):
                # img = np.random.randint(10, size=(h,w))
                if i ==numofresuts+1:
                    break

                fig.add_subplot(nrows, ncols, i)
                # plt.subplot(nrows,ncols,i+ncols)
                plt.imshow(listimages[i-1],'gray') #alpha=0.25)
                plt.title(listtitles[i-1])
                # ax[-1].set_title("ax:"+str(i-1)+listtitles[i-1])  # set title
                plt.xticks([]),plt.yticks([])

            plt.gcf().canvas.set_window_title(name_pref)
            plt.show(block=True)
            plt.pause(0.5)
            plt.close()
            """
            one can access the axes by ax[row_id][col_id]
            # do additional plotting on ax[row_id][col_id] of your choice
            ax[0][2].plot(xs, 3*ys, color='red', linewidth=3)
            ax[4][3].plot(ys**2, xs, color='green', linewidth=3)

            # ith label manifulation - change each lable value 
            label = mask[y:y + h, x:x + w] #获取要处理的方框外接矩形
            lab = label.reshape(-1, ) # 二维转一维
            lab = np.unique(lab) # 去掉重复
            lab = np.setdiff1d(lab, 0) # 去掉0值，因为0值是背景的标识，我们不用处理0，只要除去1保留2即可
            # 可以依据这个条件，获取labels = i 的坐标，并令这些坐标所在值等于0，这样labels = 2 的值并没有受到影像。
            print('len lab = ', len(lab))
            
            seed = np.argwhere(label==i)
            seedlist = list(seed)
            print('len = ', len(seedlist))
            for l in lab:
                seeds = np.argwhere(label==l)
                seedlist = list(seeds)
                print('len seedlist = ', len(seedlist))
                print(seedlist)
                if len(seedlist) == area:
                    print(seedlist)
                    
            cv2.imshow('oginal', output)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            # put all together for verification        
            """

            
            output_t = np.zeros((self.inst_image.shape[0], self.inst_image.shape[1], 3), np.uint8)           
            seed = np.argwhere(labels==lkeys)
            # label = mask[y:y + h, x:x + w] #获取要处理的方框外接矩形
            # seed = np.argwhere(label==lkeys)
            numpix = len(seed)
            for i in range(0, numpix):
                r = seed[i][0]
                c = seed[i][1]
                # r = seed[i][0]+y
                # c = seed[i][1]+x
                # seed[i][0]=r
                # seed[i][1]=c
                # output_t[r,c] =  (255, 160, 122)
                # continue
                if r<720:
                    if c<1280:
                        output_t[r,c] =  (255, 160, 122)

            # numpy_vertical = np.vstack((image, grey_3_channel))
            # numpy_horizontal = np.hstack((output, output_t))
            # plt.imshow(numpy_horizontal)         
            # plt.show(block=True)
            # plt.pause(0.5)
            # plt.close()
            # numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
            # numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
            
            # seed[:][0]=seed[:][0]+y # to avoid edge
            # seed[:][1]=seed[:][1]+x
            # print('seeds = ',seed.shape)  


            # swap by column
            # seeds[:, [0, 1]] = seeds[:, [1, 0]]
            # seeds_t= seeds.transpose()
            # swap by row
            # seeds[[0, 1],:] = seeds[[1, 0],:]
            # output = np.where(mask == seed, mask, output)
            # output_t = np.where(mask == seed, mask, output_t)
            # plt.imshow(numpy_horizontal)         
            # plt.show(block=True)
            # plt.pause(1.0)
            # plt.close()

            # output[:, :, 0][seed] = np.random.randint(0, 255)
            # output[:, :, 1][seed] = np.random.randint(0, 255)
            # output[:, :, 2][seed] = np.random.randint(0, 255)

            # output_t[:, :, 0][seed] = np.random.randint(0, 255)
            # output_t[:, :, 1][seed] = np.random.randint(0, 255)
            # output_t[:, :, 2][seed] = np.random.randint(0, 255)



            seedslist = list(seed)
            # output[:, :, 1][seeds_t] = np.random.randint(0, 123)
            sizeArea = len(seedslist)
            print('lens = ',sizeArea)  
            if sizeArea<maxarea:
                continue

            centroids_img = cv2.circle(output_t, (int(cx), int(cy)), 4, (255, 0, 255))           

            # cv2.imwrite("labels.png", put_color_to_objects(img, labels))
            # cv2.imwrite("centroids_img.png", centroids_img)
            # below need to get debugging , step by step for validation
            plots = {'Original': self.color_image, 'labels': labels, 'output': output, 'target': centroids_img}
            fig, ax = plt.subplots(1, len(plots),figsize=(16,12))
            for n, (title, im) in enumerate(plots.items()):
                cmap = plt.cm.gnuplot if n == len(plots) - 1 else plt.cm.gray
                ax[n].imshow(im, cmap=cmap)
                ax[n].axis('off')
                ax[n].set_title(title)
            # plt.figure('segments and target', figsize=(16,12))  
            plt.show(block=True)
            plt.pause(0.5)
            plt.close()
            self.grasslanes[lkeys]=rastrack(lkeys,centroids[lkeys],(x,y,w,h),seed,self.depth_filtered,self.color_filtered,occupied, empty)
            # def __init__(self, id, frame, track_pt,track_window,strawb)
            strawbs_dic[lkeys] = self.grasslanes[lkeys]
            print('lenth of refined lane_points = ',len(strawbs_dic))
            print(strawbs_dic)
        # # create figure (fig), and array of axes (ax)
        # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        """
        numofresuts = len(listimages)
        nrows = int(math.sqrt(numofresuts))
        ncols = round(numofresuts/nrows)
        print('numofresuts = ',len(listtitles))
        for i in range(1, ncols*nrows +1):#i in range(1, ncols*nrows +1):   # for i in range(1,ncols*(nrows-1)+1):
            # img = np.random.randint(10, size=(h,w))
            fig.add_subplot(nrows, ncols, i)
            # plt.subplot(nrows,ncols,i+ncols)
            plt.imshow(listimages[i-1],'gray') #alpha=0.25)
            plt.title(listtitles[i-1])
            # ax[-1].set_title("ax:"+str(i-1)+listtitles[i-1])  # set title
            plt.xticks([]),plt.yticks([])

        plt.gcf().canvas.set_window_title(name_pref)
        plt.show(block=True)
        plt.pause(0.25)
        """
        # plt.close()
        strawbs_dic = OrderedDict(sorted(strawbs_dic.items()))    
        # return self.inst_image , strawbs_dic, self.color_image
        return strawbs_dic

    def rs_callback_cvSeg_Proc(self,depth_image,rgb_image, img_inst,figsave_path=''):

        # img_ori = np.copy(rgb_image[:,:,0:3])
        # img_ori = img_ori.astype(np.uint8)
        img_ori = np.copy(self.color_image)
        # depth_image = 0.001*np.asanyarray(depth_image,dtype=np.int8)
        # color_image = cv2.cvtColor(img_ori, cv2.COLOR_RGB2BGR)

        # occupied, empty  = self.rs_callback_xyz_octo(self.depth_octo,self.color_octo,self.K,self.depth_intrin,self.depth_frame,self.num,self.figsave_path)
        # self.rgbd_image,self.depth_masks = self.rs_callback_cvSeg(depth_image,img_ori)
        # rgb_image=rgbd_image[:,:,0:3]
        # depth_image = rgbd_image[:, :, 3]
        self.rs_callback_cvSeg(depth_image,rgb_image)
        
        ################################################
        # self.depth_octo = 0.001*np.asanyarray(mapObj.depth_frame.get_data(),dtype=np.float32)
        # self.color_octo = np.asanyarray(mapObj.color_frame.get_data(),dtype=np.uint8)
        # direct using usman's output........, no working well
        tmp = np.copy(self.rgbd_image[:, :, 3])
        self.depth_octo = 0.001*np.asanyarray(tmp,dtype=np.float32)
        self.color_octo = np.asanyarray(self.rgbd_image[:,:,0:3],dtype=np.uint8)
        self.color_octo = cv2.cvtColor(self.color_octo, cv2.COLOR_BGR2RGB)

        # octomap for strawbs real data - dont use it and comment it when with open3D mapp 
        # tempD=data['depth']
        self.depth_octo[self.depth_octo > 0.5 ] = 0.0
        self.depth_octo[self.depth_octo <= 0.0] = 'nan'
        #################################################
        """
        ax3.set_title("Strawberry", fontsize=font_sz)
        ax3.imshow(depth_masks[:,:,0])
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.tick_params(axis='both', which='major', labelsize=font_sz)
        ax4.set_title("Canopy", fontsize=font_sz)
        ax4.imshow(depth_masks[:,:,1])
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.tick_params(axis='both', which='major', labelsize=font_sz)
        ax5.set_title("Rigid", fontsize=font_sz)
        ax5.imshow(depth_masks[:,:,2])
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.tick_params(axis='both', which='major', labelsize=font_sz)
        ax6.set_title("Background", fontsize=font_sz)
        ax6.imshow(depth_masks[:,:,3])
        """
        # At momment number of classs is 4
        strawbs_dic = {}
        totalClss = 4
        for id in range(0,totalClss):
        # occupied, empty  = self.rs_callback_xyz_octo(self.depth_octo,self.color_octo,self.K,self.depth_intrin,self.depth_frame,self.num,self.figsave_path)
            img_inst = self.depth_masks[:,:,id]  # for strawberry
            strawbs_dic[id] = self.rs_callback_octomapping(img_inst,img_ori)

        
        return self.inst_image,strawbs_dic,self.color_image         
           


    ##original without mathcing######################################################

     # using rosbag as input data 
    def rs_callback_cvSeg_rosbag2(self,depth_image,img_ori, img_inst,figsave_path=''):
        self.color_image = img_ori
        self.depth_image = depth_image

        plt.imshow(self.color_image)   
        plt.title('self.color_image')
        plt.show(block=False)
        plt.pause(1.0)
        plt.close()

        height, width, ch = img_inst.shape
        # img = np.zeros([height, width, 1], dtype=np.uint8)
        img = np.zeros([height, width], dtype=np.uint8)
        # info = np.iinfo(img_inst.dtype) # Get the information of the incoming image type
        # data = img_inst.astype(np.float64) / info.max # normalize the data to 0 - 1
        # data = 255 * data # Now scale by 255
        # img = data.astype(np.uint8)
        
        # _img_inst = np.copy(img_inst)
        # img[img_inst>0.1] = 0.0
        img_inst_ch = img_inst[:,:,0]
        img[img_inst_ch<0.1] = 255   
        # gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # converting to its binary form
        # ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
        kernel = np.ones((9, 9), 'uint8')
        self.inst_image = cv2.dilate(img, kernel, iterations=1)

        # numpy_vertical = np.vstack((image, grey_3_channel))
        numpy_horizontal = np.hstack((img, self.inst_image))

        # numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
        # numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
        cv2.imshow('Ori-Dilated Image', numpy_horizontal)
        cv2.waitKey(500)
        cv2.destroyAllWindows()      

        recxy = 10     
        h, w, ch = self.color_image.shape
        dim = (w,h)
        self.inst_image = cv2.resize(self.inst_image,dim, interpolation=cv2.INTER_LINEAR)   

        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(self.color_image,cmap='gray')
        axarr[1].imshow(self.inst_image,cmap='gray')

        name = 'rsiz_inst_enhance_'+str(self.num)+'.png' 
        sved_inst = figsave_path+name   
        plt.savefig(sved_inst, bbox_inches='tight')     
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()    

        self.frame, centers,det_store = self.rsDetTrck.detectMore_trackOneNN(
            self.color_image, self.inst_image,5000, debugMode=True, name_pref='strawbs detect')
        
        # strawbs detection
        name = 'straw_det_' + str(self.num)+'.png'
        sved_det = figsave_path+name  
        plt.imshow(self.frame)   
        plt.savefig(sved_det, bbox_inches='tight')     
        plt.show(block=False)
        plt.pause(1.0)
        plt.close()
        # option for Rob to choose  - list ,dictionary  adapted into saga's tracking system
        for id, item in det_store.items():   # ditctionary = [centrr , and [w, h]]
            print(item)
            print(det_store[id])

        det_store_lst = list(sorted(det_store.keys()))
        print(det_store_lst)

        src2 = np.zeros_like(self.color_image)
        src2[:,:,0] = self.inst_image[:,:]
        src2[:,:,1] = self.inst_image[:,:]
        src2[:,:,2] = self.inst_image[:,:]
        self.inst_image = src2

        depth_points = []
        for count, value in enumerate(centers):
            print(count, value)
            cx = int(centers[count][0])
            cy = int(centers[count][1])
            if cx < recxy or cy < recxy:
                break
            #################################################################################
            # 3D translation and transformation with centroilds, realsense. open3d
            #################################################################################
            ix = int(cx)
            iy = int(cy)
            # just for the verification of fomular - seems agree to each other
            depth = self.depth_scale*self.depth_image[iy, ix]
            # depth = self.depth_frame.get_distance(ix, iy)
            depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [ix, iy], depth[0])
            print ('result:', depth_point)
            # sys.stdout.write('%s: Depth at center(%d, %d): %f(mm)\r' % (self.topic, pix[0], pix[1], cv_image[pix[1], pix[0]]))
            # sys.stdout.flush()
            depth_points = []

            # Detect object maxctroid: the largest contour
            # distance, depth_point = self.rsDetTrck.Pos2DPixels3Dxyz(
            #     self.depth_frame, cx, cy, self.color_intrin, self.depth_intrin, self.depth_scale)
            if depth_point is not None:
                
                depth_points.append(depth_point)

                text = "xyz: %.5lf, %.5lf, %.5lfm" % (
                    depth_point[0], depth_point[1], depth_point[2])
                # cv2.putText(color_frame, "Measured xyz{}m".format(distance), (depth_point[0], depth_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                # cv2.putText(frame, text, (int(depth_point[0]), int(depth_point[1]) - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                cv2.putText(self.color_image, text, (cx + recxy,
                            cy - recxy), 0, 0.5, (0, 200, 255), 2)
                cv2.circle(self.color_image, (cx, cy), 8, (0, 0, 255), 1)
                # drawn on colormap - acting as mesh at momoment:

                cv2.putText(self.frame, text,(cx + recxy, cy - recxy), 0, 0.5, (0, 200, 255), 2)
                cv2.circle(self.frame,(cx, cy), 8, (0, 0, 255), 1)

                cv2.putText(self.inst_image, text,(cx + recxy, cy - recxy), 0, 0.5, (0, 200, 255), 2)
                cv2.circle(self.inst_image,(cx, cy), 8, (0, 0, 255), 1)

                """figname = 'strw_xyz_frm'+str(self.num)+'.png'
                # os.chdir(OUTDIR)
                nampepath = os.path.join(output_path, figname)
                depth_colormap = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(nampepath, depth_colormap)"""
                # Stack both images horizontally
                # images = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_colormap))
                # os.chdir("..")  

        figname = 'strw_xyz_frm'+str(self.num)+'.png'
        # os.chdir(OUTDIR)
        nampepath = os.path.join(self.output_path, figname)
        depth_colormap = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(nampepath, depth_colormap)

        # avi_res = np.hstack((depth_colormap, self.frame))

        avi_res = np.hstack((depth_colormap, src2))
        avi_res = cv2.resize(avi_res,(self.avi_width,self.avi_height), interpolation=cv2.INTER_LINEAR) 
        cv2.imshow('avi_res', avi_res)
        
        key = cv2.waitKey(500)
        self.outRes.write(avi_res)
        if key & 0xFF == ord('s'):
            cv2.waitKey(0)
        if key==27 & 0xFF == ord('q'):    # Esc key to stop, 113: q
            cv2.destroyAllWindows()
            return
        
        numpy_horizontal
        return centers, depth_points
    ##################################################################################
    ##################################################################################

    def rs_callback_cvSeg_bag(self):
        # data = imgviz.data.arc2017()
        # camera_info = data['camera_info']
        # K = np.array(camera_info['K']).reshape(3, 3)
        # rgb = data['rgb']
        
        # depth_image = data['depth']
        #  https://github.com/IntelRealSense/realsense-ros/issues/1342

        recxy = 10

        centers, frame = self.rsDetTrck.detectMore_trackOne(
            self.color_image, self.lowerLimit, self.upperLimit, 800, debugMode=True, name_pref='strawbs detect')

        for count, value in enumerate(centers):
            print(count, value)
            cx = int(centers[count][0])
            cy = int(centers[count][1])
            if cx < recxy or cy < recxy:
                break
            #################################################################################
            # 3D translation and transformation with centroilds, realsense. open3d
            #################################################################################
            # Detect object maxctroid: the largest contour
            distance, depth_point = self.rsDetTrck.Pos2DPixels3Dxyz(
                self.depth_frame, cx, cy, self.color_intrin, self.depth_intrin, self.depth_scale)
            text = "xyz: %.5lf, %.5lf, %.5lfm" % (
                depth_point[0], depth_point[1], depth_point[2])
            # cv2.putText(color_frame, "Measured xyz{}m".format(distance), (depth_point[0], depth_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            # cv2.putText(frame, text, (int(depth_point[0]), int(depth_point[1]) - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            cv2.putText(self.color_image, text, (cx + recxy,
                        cy - recxy), 0, 0.5, (0, 200, 255), 2)
            cv2.circle(self.color_image, (cx, cy), 8, (0, 0, 255), 1)
            # drawn on colormap - acting as mesh at momoment:

            cv2.putText(self.depth_colormap_aligned, text,
                        (cx + recxy, cy - recxy), 0, 0.5, (0, 200, 255), 2)
            cv2.circle(self.depth_colormap_aligned,
                       (cx, cy), 8, (0, 0, 255), 1)

            figname = 'strw_xyz_msh'+str(self.num)+'.png'
            # os.chdir(OUTDIR)
            nampepath = os.path.join(self.output_path, figname)
            # depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
            cv2.imwrite(nampepath, self.color_image)
            # Stack both images horizontally
            # images = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_colormap))
            # os.chdir("..")

        distance, depth_point = self.rsDetTrck.Pos2DPixels3Dxyz(
            self.depth_frame, cx, cy, self.color_intrin, self.depth_intrin, self.depth_scale)
        text = "xyz: %.5lf, %.5lf, %.5lfm" % (
            depth_point[0], depth_point[1], depth_point[2])
        # cv2.putText(color_frame, "Measured xyz{}m".format(distance), (depth_point[0], depth_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        # cv2.putText(frame, text, (int(depth_point[0]), int(depth_point[1]) - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        cv2.putText(self.color_image, text, (cx + recxy,
                    cy - recxy), 0, 0.5, (0, 200, 255), 2)
        cv2.circle(self.color_image, (cx, cy), 8, (0, 0, 255), 1)
        # drawn on colormap - acting as mesh at momoment:
        # cv2.putText(self.depth_colormap_aligned,text, (cx + recxy, cy - recxy), 0, 0.5, (0,200,255), 2)
        # cv2.circle(self.depth_colormap_aligned, (cx,cy), 8, (0, 0, 255), 1)

        figname = 'strw_xyz_pos'+str(self.num)+'.png'
        # os.chdir(OUTDIR)
        nampepath = os.path.join(self.output_path, figname)
        # depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
        cv2.imwrite(nampepath, self.color_image)      

        # pcd = pointcloud_from_depth(
        #     self.depth_image, fx=self.fx, fy=self.fy, cx=self.ppx, cy=self.ppy)

        return centers, depth_point
    def rs_callback_cvSeg_rosbag(self):
        recxy = 10
        centers, frame = self.rsDetTrck.detectMore_trackOne(
            self.color_image, self.lowerLimit, self.upperLimit, 800, debugMode=True, name_pref='strawbs detect')

        for count, value in enumerate(centers):
            print(count, value)
            cx = int(centers[count][0])
            cy = int(centers[count][1])
            if cx < recxy or cy < recxy:
                break
            #################################################################################
            # 3D translation and transformation with centroilds, realsense. open3d
            #################################################################################
            # Detect object maxctroid: the largest contour
            distance, depth_point = self.rsDetTrck.Pos2DPixels3Dxyz(
                self.depth_frame, cx, cy, self.color_intrin, self.depth_intrin, self.depth_scale)
            text = "xyz: %.5lf, %.5lf, %.5lfm" % (
                depth_point[0], depth_point[1], depth_point[2])
            # cv2.putText(color_frame, "Measured xyz{}m".format(distance), (depth_point[0], depth_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            # cv2.putText(frame, text, (int(depth_point[0]), int(depth_point[1]) - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            cv2.putText(self.color_image, text, (cx + recxy,
                        cy - recxy), 0, 0.5, (0, 200, 255), 2)
            cv2.circle(self.color_image, (cx, cy), 8, (0, 0, 255), 1)
            # drawn on colormap - acting as mesh at momoment:

            cv2.putText(self.depth_colormap_aligned, text,
                        (cx + recxy, cy - recxy), 0, 0.5, (0, 200, 255), 2)
            cv2.circle(self.depth_colormap_aligned,
                       (cx, cy), 8, (0, 0, 255), 1)

            figname = 'strw_xyz_msh'+str(self.num)+'.png'
            # os.chdir(OUTDIR)
            nampepath = os.path.join(self.output_path, figname)
            # depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
            cv2.imwrite(nampepath, self.color_image)
            # Stack both images horizontally
            # images = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_colormap))
            # os.chdir("..")

        distance, depth_point = self.rsDetTrck.Pos2DPixels3Dxyz(
            self.depth_frame, cx, cy, self.color_intrin, self.depth_intrin, self.depth_scale)
        text = "xyz: %.5lf, %.5lf, %.5lfm" % (
            depth_point[0], depth_point[1], depth_point[2])
        # cv2.putText(color_frame, "Measured xyz{}m".format(distance), (depth_point[0], depth_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        # cv2.putText(frame, text, (int(depth_point[0]), int(depth_point[1]) - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        cv2.putText(self.color_image, text, (cx + recxy,
                    cy - recxy), 0, 0.5, (0, 200, 255), 2)
        cv2.circle(self.color_image, (cx, cy), 8, (0, 0, 255), 1)
        # drawn on colormap - acting as mesh at momoment:
        # cv2.putText(self.depth_colormap_aligned,text, (cx + recxy, cy - recxy), 0, 0.5, (0,200,255), 2)
        # cv2.circle(self.depth_colormap_aligned, (cx,cy), 8, (0, 0, 255), 1)

        figname = 'strw_xyz_pos'+str(self.num)+'.png'
        # os.chdir(OUTDIR)
        nampepath = os.path.join(self.output_path, figname)
        # depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
        cv2.imwrite(nampepath, self.color_image)     

        # pcd = pointcloud_from_depth(
        #     self.depth_image, fx=self.fx, fy=self.fy, cx=self.ppx, cy=self.ppy)

        return centers, depth_point     