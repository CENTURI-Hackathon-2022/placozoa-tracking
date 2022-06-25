import pandas as pd
import cv2
from datetime import timedelta, datetime
import re
import os
import numpy as np


def csv_name_creator(path, output_folder='C:\\Users\\domin\\Documents\\SCHOOL\\hackathon\\placozoa-tracking_Domi\\data'):
    """
    path: folder + file_name of the video
    output_folder: where the csv files should be stored
    returns :
        - a string path as input for pandas 'to_csv()' fun
        - timestamp of the exact date&time of the beginning of the recording
    """

    # extract name of the video
    try:
        found = re.search('Basler_(.+?).mp4', path).group(1)
    except AttributeError:
        # AAA, ZZZ not found in the original string
        found = ''  # apply your error handling------------------------

    try:
        date = re.search('__[0-9]+__(.+?)[0-9]{3}.mp4', path).group(1)
    except AttributeError:
        # AAA, ZZZ not found in the original string
        date = ''  # apply your error handling------------------------

    millisec = int(re.findall('(\d{3})(?!.*\d)', found)[0])

    # check whether the folder where you want to save it in exist, if not create it
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # new path
    new_path = os.path.join(output_folder, found + ".csv")

    # Date-time beginning of timestamp
    # first_date=pd.to_datetime(''.join(re.findall('[0-9]+', date)))
    first_date = ''.join(re.findall('[0-9]+', date))
    first_stamp = pd.Timestamp(first_date) + timedelta(milliseconds=millisec)

    return new_path, first_stamp

def tif_name_creator(path, output_folder="/media/data-119/rat596_20210701_184333", rat="rat1"):
    """path: folder+ file_name of the video
    output_folder: where all the npy files should be stored
    return:- a string path as input for np.save() fun"""

    # extract name of the video
    try:
        found = re.search('Basler_(.+?).mp4', path).group(1)
    except AttributeError:
        # AAA, ZZZ not found in the original string
        found = ''  # apply your error handling------------------------

    # check whether the folder where you want to save it in exist, if not create it
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # new path
    new_path = os.path.join(output_folder, found + "_{}.npy".format(rat))

    return new_path


# Frames per second counter
class FPS:
    def __init__(self):
        """store the start time, end time, and total number of frames
        that were examined between the start and end intervals"""

        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


def resizing(frame, scale_percent=50):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized


