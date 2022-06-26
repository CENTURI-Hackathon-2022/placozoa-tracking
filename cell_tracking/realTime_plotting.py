# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 23:24:36 2022

@author: domin
"""

import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
from helper_funs import csv_name_creator, FPS, resizing, timestamping, timedelta, npy_name_creator
#from getting_coordinates_manual_CLASS import Rat_coords
#from realTime_plotting_CLASS import Rat_mvm
import matplotlib.pyplot as plt
import json
import path
import os


#define plotting class
class Viewer:
    '''create object 'rat_movement':
        - will be updated for every frame
        - plot updated plot'''
        
    def __init__(self):
        #init figure
        self.xs = []  #millisecs converted to secs
        self.ys1 = [] #rat 1
        self.ys2 = [] #rat2
        
        self.fig = plt.figure(figsize=(25, 25))
        self.gs = self.fig.add_gridspec(1, 1)
    
        self.ax1 = self.fig.add_subplot(self.gs[0, 0])
                
   
    
    def update(self, frame): #m_rat1, m_rat2, fps_vs, secs):
        
        #tight layout
        #plt.tight_layout()
        
        #axes off
        for ax in [self.ax1]:#,self.ax4,self.ax5]:
            ax.set_axis_off()
        
        #VIDEO
        self.ax1.imshow(frame)
        
        plt.show()
        plt.autoscale()
        plt.pause(0.00000000000001)
        #clear axis
        self.ax1.clear()

def plot_brightCells(path2vid = "C://Users//domin//Documents//SCHOOL//hackathon//placozoa-tracking_Domi//data/film1.avi",
                      scale_percent = 70,
                      fs = 20,
                      delta_thresh = 25,#pixel intensity difference threshold
                      play_vid = True,
                      save_npy = False,
                      npyName = ""):
                    
    # read from a video file & get video parameter FramePerSec (FPS)
    vs = cv2.VideoCapture(path2vid)
    
    #play what you've done
    if play_vid:
        #init Viewer object
        cm = Viewer()
        
    #init test_frame_list 
    test_frame_list = []
    avgframe = []
    
    # loop over the frames of the video
    n = 0
    #while True:
    for n in range(50): #5sec
        ret, frame = vs.read()  # ret says whether the frame exists
    
        if frame is None:
            break  # end of video if no more frames
       
        
        # resize the frame, convert it to grayscale, and blur it        
        frame = resizing(frame=frame, scale_percent=scale_percent)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        #for i in np.logspace(start=1, stop = 2,  num=120, endpoint=True, base=2, dtype=None, axis=0):
            #new_image(i,j) = alpha*image(i,j) + beta
        gray = cv2.convertScaleAbs(gray, alpha=5.5, beta=3) #contrast (multiplying pixel intensities by a factor)
        #gray = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT) #useleff cause we need detail
        # initialize average frame (=background)
        if len(avgframe) == 0:
                avgframe = frame.copy().astype("float")
    
        # update average & calculate difference between current and running avg
        cv2.accumulateWeighted(frame, avgframe, 0.5) 
        frameDelta = cv2.absdiff(frame,
                                    cv2.convertScaleAbs(avgframe))
    
        # threshold the delta image, dilate the thresholded image to increase size of dots
        thresh = cv2.threshold(frameDelta, delta_thresh, 255,
                                  cv2.THRESH_BINARY)[1]
        #DIFFERENCE dileted vs. non dilated: non dilated more detailed and takes less time
        thresh = cv2.dilate(thresh, None, iterations=4)  # can spread the white pixels with itterations.. 
        tr = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY) #otherwise it save as BGR
        test_frame_list.append(tr)

        if play_vid: 
            cm.update(tr)
        if save_npy:
            #save treshed frames in npy
            np.save(os.path.join(os.path.split(path2vid)[0],npyName) ,test_frame_list)
    plt.close()
    return test_frame_list


def plot_Trace(path2thresh = "C:/Users/domin/Documents/SCHOOL/hackathon/placozoa-tracking_Domi/data/thresh_less_dilated.npy",
        play_vid = True):
    
    #thresh_smaller = np.load()
    g_smaller = np.load(path2thresh)
    g2plot_list = []
    #plot TRACE (pixel value addition)
    
    g2plot = g_smaller[0]
    #play what you've done
    if play_vid:
        #init Viewer object
        cm = Viewer()
    for i in range(1,len(g_smaller)):
        g2plot = (g2plot/1.01)+g_smaller[i]*1000
        g2plot_list.append(g2plot)
        if play_vid: 
            cm.update(cv2.dilate(g2plot, None, iterations=1))
        #plt.imshow(g2plot, aspect='auto', cmap=plt.get_cmap('plasma'))
        #plt.show()
    plt.close()
    #np.save("C:\\Users\\domin\\Documents\\SCHOOL\\hackathon\\placozoa-tracking_Domi\\data\\summed.npy", g2plot_list)


def plot_denseOpticFlow(g_smaller,scale_percent= 30):
    
    cap = cv2.VideoCapture("C://Users//domin//Documents//SCHOOL//hackathon//placozoa-tracking_Domi//data/film1.avi")
    ret, frame1 = cap.read()
    frame1 = resizing(frame=frame1, scale_percent=scale_percent)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(cv2.cvtColor(g_smaller[1], cv2.COLOR_GRAY2BGR)) #needs to be RGB
   
    hsv[..., 1] = 255
    #hsv = g_smaller[1]
    while(1):
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame2 = resizing(frame=frame2, scale_percent=scale_percent)
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.4, 3, 3, 2, 3, 1.5, 2)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2', bgr)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
       # elif k == ord('s'):
            # cv.imwrite('opticalfb.png', frame2)
            # cv.imwrite('opticalhsv.png', bgr)
        prvs = next
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # g_smaller = np.load("C:/Users/domin/Documents/SCHOOL/hackathon/placozoa-tracking_Domi/data/thresh_less_dilated.npy")
    # g_smaller = plot_brightCells(path2vid = "C://Users//domin//Documents//SCHOOL//hackathon//placozoa-tracking_Domi//data/film1.avi",
    #                               scale_percent=50, 
    #                               save_npy=False,
    #                               npyName = "C:/Users/domin/Documents/SCHOOL/hackathon/placozoa-tracking_Domi/data/thresh_less_dilated.npy")
    # plot_Trace(path2thresh = "C:/Users/domin/Documents/SCHOOL/hackathon/placozoa-tracking_Domi/data/thresh_less_dilated.npy",
    #         play_vid = True)
    plot_denseOpticFlow(g_smaller, scale_percent=50)