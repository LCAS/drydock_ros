from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

print(__doc__)

# python modules
import os
import sys
import time


# third party modules
import cv2
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import pyrealsense2 as rs


# import our own modules
from dom_octo_mapper.tools_dl.rs_insertPointCloudSegPtoch_usmanl_octo_3 import MapManager


device = torch.device(0)


matplotlib.use('TkAgg')
print('sys version: ', sys.version)
print('torch version: ', torch.__version__)
print ('cv2 version:', cv2.__version__)
curpth = os.path.abspath(os.getcwd())
abcurpth = os.path.dirname(os.path.abspath(__file__))
print('pwd:{pth} and absoute path {abpth}'.format(pth = curpth, abpth = abcurpth))

WIDTH = 1280
HEIGHT = 720
torch.backends.cudnn.enabled = True
cudaflag = torch.cuda.is_available()
print('torch.cuda availabe ?:' ,cudaflag)
device = torch.device('cuda:0' if cudaflag else 'cpu')
print(device)
isWebCam = False # using webcam for .avi
# output_path = 'results'       

# Run only if this module is being run directly
def main():

    # processing all data there
    mapObj = MapManager()
    vis_size = (mapObj.avi_width,mapObj.avi_height)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    svRctRes=os.path.join(mapObj.figsave_path, 'ras_pcetion_'+timestr+'.avi')
    mapObj.outRes = cv2.VideoWriter(svRctRes,mapObj.fourcc2, 10.0,vis_size)

    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    if isWebCam==True:
        # Create opencv video capture object
         VideoCap = cv2.VideoCapture('dataset/2014_08_04__15_48_17.avi')
       
    else:
  
        
        mapObj.bagfileFlg=True       
        for x in range(5):
            mapObj.pipeline.wait_for_frames()
    while True:
        if isWebCam==True:
            ret, color_image = VideoCap.read()  
            # img_ori = cv2.imread(dataset_dir)   
            img_ori = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)              
        else:          
        
            # for attension of Jonann: this is where to access Usman output starting entrance, but inside , is the local acess to usman's codes
            # I belive you are genius on how to handle this under your structure.-  good luck ot use it , explore..,
           
            # for using Usman's interface
            # rgb_usman = color_image.astype(np.int8)
            # depth_usman = depth_image.astype(np.int8)
            img_inst = None
            depth_usman_3ch = None
            rgb_usman = None
            img_inst = None
            img_inst, strawbs_dic,seg_image = mapObj.rs_callback_cvSeg_Proc(depth_usman_3ch,rgb_usman,img_inst)
            ############optization by data fusiong, - data assocation mathcing ###########
                        
            # output = np.zeros((img_ori.shape[0], img_ori.shape[1], 3), np.uint8) 
            output_t = np.zeros((img_ori.shape[0], img_ori.shape[1], 3), np.uint8)  
            count_id = -1 
            for ikey, pt in strawbs_dic.items():#.iteritems(): 
                id = pt.id # label number - recording identity
                x,y,w,h = pt.track_win # track_window (x,y,w,h)
                center = pt.center # tracking point - centroid of roi
                cx = int(center[0])
                cy = int(center[1])
                roi = pt.roi #frame[y-h:y, x-w:x]#cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)        
                print('id = ',id) 
                print('win = ',x,y,w,h) 
                print('depth_points = ',pt.depth_points) 

                print('center cx, cy = ',center[0], center[1])                 
                print('roi size = ',len(roi))
                # calculate depth vector and 3D vector in another way
                cy_vec = roi[:,0]
                cx_vec = roi[:,1]
                print('cy_vec = ',cy_vec) 
                print('cx_vec = ',cx_vec)            
                #####################
                depth_filtered = pt.depth_obj # keep individual segmented obj depth image
                color_filtered = pt.color_obj # hold individual segmented obj color image
                plots = {'Ori-RGB': mapObj.color_image, 'Ori-depth': mapObj.depth_image, 'depth_obj': depth_filtered, 'color_obj': color_filtered}
                fig, ax = plt.subplots(1, len(plots), figsize=(16,12))
                for n, (title, im) in enumerate(plots.items()):
                    cmap = plt.cm.gnuplot if n == len(plots) - 1 else plt.cm.gray
                    ax[n].imshow(im, cmap=cmap)
                    ax[n].axis('off')
                    ax[n].set_title(title)
                # plt.figure('segments and target', figsize=(16,12)) 
                
                plt.gcf().canvas.set_window_title('Overall and Individual Obj')
                count_id = count_id+1 
                name = 'all_obj_sum_'+str(mapObj.num)+"_"+str(count_id)+'.png' 
                sved_inst = mapObj.figsave_path+name   
                plt.savefig(sved_inst, bbox_inches='tight')     
                plt.show(block=True)
                plt.pause(0.5)
                plt.close()  


                full_mask = np.zeros( img_ori.shape[:2] , dtype=np.uint8)
                full_mask[full_mask==roi]=255
                # seed = np.argwhere(full_mask==255)
                indices =(np.array(full_mask)).reshape( (-1, int( len(full_mask)) ) )
                indices = tuple(map(tuple, indices))

                seed = np.where(full_mask == 255)
                # full_mask = np.where(full_mask == r, mask, output)
                # output_t = np.where(mask == seed, mask, output_t)
                # deth_clmap = np.hstack((color_image, depth_colormap))
                print(seed)
                plt.imshow(full_mask, cmap='gray')
                plt.show(block=False)
                plt.pause(0.5)
                plt.close()               

                depth = mapObj.depth_frame.get_distance(cx, cy)
                pt.depth_points = rs.rs2_deproject_pixel_to_point(mapObj.color_intrin, [cx, cy], depth)
                print ('depth_point:', pt.depth_points)              

            ##############################################################################    

         


if __name__ == '__main__':
    main()