# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 21:30:41 2022

@author: domin
"""

import cv2
import numpy as np
import pandas as pd
from motion_detector import csv_name_creator, FPS, resizing, timestamping, timedelta, npy_name_creator
#from getting_coordinates_manual_CLASS import Rat_coords
#from realTime_plotting_CLASS import Rat_mvm
import matplotlib.pyplot as plt
import json
import path
import os

path2vid = "C:\\Users\\domin\\Documents\\SCHOOL\\hackathon\\placozoa-tracking_Domi\\data"

def cell_tracker(co, 
                              fw, fh, #need this for Writer object
                              path, 
                              #number_of_rats, #defined in getting_manual_coordinates_CLASS.py
                              #coordinates, frame_width, frame_height,
                              #rm, #rat movememnt calss object
                              scale_percent=50, 
                              area=20, 
                              delta_thresh=5,
                              plotting_realTime = False, #set this on if you wanna see the processing
                              save_npy = True,
                    output_4csv="C:\\Users\\domin\\Documents\\SCHOOL\\STAGE2\\motion_detection_4ephys\\data\\CSV_outputs",
                    output_4npy="C:\\Users\\domin\\Documents\\SCHOOL\\STAGE2\\motion_detection_4ephys\\data\\NPY_outputs"):
                   # output_rat1_mp4="C:/Users/domin/Documents/SCHOOL/STAGE2/motion_detection_4ephys/data/video_outputs/test_rat1{}".format(format_output),
                   # output_rat2_mp4="C:/Users/domin/Documents/SCHOOL/STAGE2/motion_detection_4ephys/data/video_outputs/test_rat2.{}".format(format_output)):
    """
    IMPORTANT: when selecting rectagles for rats -> rat 1 is on the left on the picture
    path: path to the video (this is defined in getting_manual_coordinates_CLASS.py)
    output_4csv: collects timestamp (date&time + sec count), frame number, movement 
    output npy: used to save json coordinates. If you're interested in saving processed frames, enable save_npy.
    """
    
    #if specified output folder doesn't exist, create it
    if not os.path.exists(output_4csv):
        os.mkdir(output_4csv)
    
    #using rc object attributes for local vars
    coordinates = co
    if plotting_realTime:
        cm = Rat_mvm()
    #fw and fh needed for Writer object
    frame_width = fw
    frame_height = fh
        
    # read from a video file & get video parameter FramePerSec (FPS)
    vs = cv2.VideoCapture(path)
    fps_vs = vs.get(cv2.CAP_PROP_FPS)
    #print("Estimated frame rate of the file: {}".format(fps_vs))
    
    #create path to save json with coordinates (and npy if save_npy)
    newpath_npy = npy_name_creator(path, output_folder=output_4npy, rat="all_rats")
    #carful.. vs.get(cv2.CAP_PROP_FRAME_COUNT) not accurate!
    
    #WRITE BIG ARRAY WITH NP.MEMMAP()
    if save_npy:
        #crete mapping object to save big array#-----------------------------------------------------------------memmap
        npyMap = np.memmap(newpath_npy, dtype='uint8', mode='w+', 
                           shape=(2, int(vs.get(cv2.CAP_PROP_FRAME_COUNT)), frame_height, frame_width))#-----------------
    
    #store shape of npy array into metada for reload
    json_str = {
        "shape" : [2, int(vs.get(cv2.CAP_PROP_FRAME_COUNT)), frame_height, frame_width],
        "coordinates" : coordinates #you can plot it later
        }
    print("Expected n of frames: {}".format(vs.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    #initiate PROCESSING FramePerSec (FPS) count
    fps = FPS().start()
    
    # initialize average frame, frame delta and thresholded frame
    avgframe = [[], []]
    frameDelta = [[], []]
    thresh = [[], []]
    #if save_npy:

    # initiate dictionary
    ts_dict = {"frame_nb": [], "millisec": [], "mvm_rat1": [], "mvm_rat2": []}
    
    # loop over the frames of the video
    n = 0
    while True:
        ret, frame = vs.read()  # ret says whether the frame exists
        
        if frame is None:
            break  # end of video if no more frames

        # get timestamp of each frame...------------------------------------------------------------------------
        #millisec = str(vs.get(cv2.CAP_PROP_POS_MSEC))... it's outputting 0 in tail of video.. ??---------------
        # in millisecond
        millisec = (float(n)/fps_vs)*1000    
        ts_dict = timestamping(ts_dict=ts_dict,
                               frame_nb=n,
                               millisec=millisec)
        
        # resize the frame, convert it to grayscale, and blur it        
        frame = resizing(frame=frame, scale_percent=scale_percent)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT) #blur kernel must be odd number, with bigger blur you lose thin shapes
        
        #init mask for both rats
        mask_rat = [np.zeros(gray.shape, dtype="uint8"),
                    np.zeros(gray.shape, dtype="uint8")]
        
        
        #crop 'masks' with coordinates 
        cv2.rectangle(mask_rat[0], coordinates[0], coordinates[1], 255, -1)
        cv2.rectangle(mask_rat[1], coordinates[2], coordinates[3], 255, -1)

        
        # itter over rats
        for i in range(number_of_rats): #default = 2
            #apply cropped mask with bitwise apperation
            tmp_rat_sections = cv2.bitwise_and(gray, gray, mask=mask_rat[i])

            # initialize average frame (=background)
            if len(avgframe[i]) == 0:
                    avgframe[i] = tmp_rat_sections.copy().astype("float")

            # update average & calculate difference between current and running avg
            cv2.accumulateWeighted(tmp_rat_sections, avgframe[i], 0.1) #tried: 0.2
            frameDelta[i] = cv2.absdiff(tmp_rat_sections,
                                        cv2.convertScaleAbs(avgframe[i]))

            # threshold the delta image, dilate the thresholded image to fill
            # TO DO:
            # in holes, then find contours on thresholded image... 
            thresh[i] = cv2.threshold(frameDelta[i], delta_thresh, 255,
                                      cv2.THRESH_BINARY)[1]
            #DIFFERENCE dileted vs. non dilated: non dilated more detailed and takes less time
            thresh[i] = cv2.dilate(thresh[i], None, iterations=1)  # can spread the white pixels with itterations.. 
            
            
        # append mvm count for each rat
        for i in range(number_of_rats): #default = 2
            ts_dict[f"mvm_rat{i + 1}"].append(np.sum(thresh[i]))
            
            if save_npy:
                #write into memmaped npy#--------------------------------------------------------------------------------memmap
                npyMap[i][n] = thresh[i]#-------------------------------------------------------------------------------------
        
        # update FPS counter
        fps.update()
        
        if plotting_realTime:
            #Animated plotting
            rm.update(ts_dict = ts_dict, fps_vs=fps_vs, secs = 5)
         
        #itter frame count
        n += 1
        if n == 100:
            print("Processed 100 frames and still processing...")
        elif n == 10000:
            print("Processed 10000 frames and still processing...")
        
        if plotting_realTime:
            # DISPLAY VIDEOS (IT'S at least 2x LONGER THIS WAY... 92 vs. 177 FPS)
            cv2.imshow("webcam", frame)
            cv2.imshow("Thresh_rat1", thresh[0])
            cv2.imshow("Thresh_rat2", thresh[1])
                    
            #keep open
            #cv2.waitKey(int(1000/fps_vs)) #if you want it be displayed in normal speed.. but with plotting it gets super slow
            cv2.waitKey(1)

        
    # stop the timer and print Frame Per Sec info
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print("Processed number of frames: {}".format(n))

    # CLEAN UP videoCapture * videoWriter objects
    vs.release()
    if save_npy:
        del npyMap #deletion flushes memory changes to disk before removing the object#--------------------- memmap
    if plotting_realTime:
        plt.close('all')
        cv2.destroyAllWindows()

    # get path for csv output & init of timestamp (extracted from the video filename)
    newpath_csv, first_stamp = csv_name_creator(path, output_folder=output_4csv)
    # newpath_npy_rat1 = npy_name_creator(path, output_folder=output_4npy, rat="rat1")-------------------
    
    # convert dict do pandas data frame...
    ts_df = pd.DataFrame(ts_dict)

    # create column 'duration' (millisec converted to time)
    ts_df['duration'] = [timedelta(milliseconds=float(i)) for i in ts_df['millisec']]
    # to keep only the time of the timedelta (otherwise give you estimate with days as well)
    ts_df['duration'] = ts_df['duration'].values.astype('datetime64[ns]')
    ts_df['duration'] = [i.strftime("%H:%M:%S.%f")[:-3] for i in ts_df['duration']]

    # ...set exact datetime timestamp (adding the first stamp to milliseconds)...
    ts_df['timestamp'] = [pd.to_timedelta(str(i) + 'millisecond') +
                          first_stamp for i in ts_df['millisec']]
    ts_df.set_index('timestamp', inplace=True)

    order = [0, 1, 4, 2, 3]  # setting columns' order
    ts_df = ts_df[[ts_df.columns[i] for i in order]]
    
    #Normalize movement
    ts_df['mvmr1_normalized'] = [(x - np.min(ts_df["mvm_rat1"]))/(np.max(ts_df["mvm_rat1"]) - np.min(ts_df["mvm_rat1"])) 
                                 for x in ts_df['mvm_rat1']]
    ts_df['mvmr2_normalized'] = [(x - np.min(ts_df["mvm_rat2"]))/(np.max(ts_df["mvm_rat2"]) - np.min(ts_df["mvm_rat2"])) 
                                 for x in ts_df['mvm_rat2']]

    # ...and save it to info directory
    ts_df.to_csv(newpath_csv, sep=',')
    # return ts_df
    
    #write JSON
    with open(newpath_npy[:-4]+".json", "w") as outjson:
        json.dump(json_str, outjson)

#if __name__ == '__main__':
#    #path to test video
#    path2vid_test="C:/Users/.../motion_detection_4ephys/data/Basler_acA1300-60gmNIR__21471690__20211207_113623925_SHORT.mp4"
#    motion_detector_MVMwriter(path2vid_test)