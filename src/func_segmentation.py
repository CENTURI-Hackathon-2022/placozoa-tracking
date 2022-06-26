import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import morphsnakes as ms
import napari
import numpy as np
import pandas as pd
import scipy.ndimage as nd
import skimage
from numpy.typing import ArrayLike
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_fill_holes
from skimage import exposure, filters, morphology
from skimage.draw import ellipse
from skimage.filters import rank
from skimage.measure import label, perimeter, regionprops, regionprops_table
from skimage.morphology import (convex_hull_image, disk, remove_small_holes,
                                remove_small_objects)
from skimage.segmentation import clear_border
from skimage.transform import rotate
from tifffile import imread, imwrite
from yapic.session import Session


def segmentation_chanvese(image:ArrayLike,
                        disk_size:int=4,
                        iteration_nb:int=10) -> ArrayLike:
    image = np.array(image)

    output_array = np.zeros(image.shape, dtype = bool)

    for t in range(0,image.shape[0]):

        im_single_t = image[t,:,:]
        im_filtered_minimum =  rank.minimum(im_single_t, disk(disk_size))
        im_ms = ms.morphological_chan_vese(im_filtered_minimum, iteration_nb)
        ms_filled = binary_fill_holes(im_ms)

        #detect if its is segmented the right way around (expecting that the background has most area touching the image border)
        #otherwise invert the image
        amount_edge_false = ms_filled[ms_filled[0,:] == False].shape[0] + ms_filled[ms_filled[-1,:] == False].shape[0] + ms_filled[ms_filled[:,0] == False].shape[0] + ms_filled[ms_filled[:,-1] == False].shape[0]
        amount_edge_true = ms_filled[ms_filled[0,:] == True].shape[0] + ms_filled[ms_filled[-1,:] == True].shape[0] + ms_filled[ms_filled[:,0] == True].shape[0] + ms_filled[ms_filled[:,-1] == True].shape[0]
        if amount_edge_true < amount_edge_false:
            pass
        else:
            ms_filled = np.invert(ms_filled)

        #label connected components in the binary mask
        labels, num_features = nd.label(ms_filled)
        label_unique = np.unique(labels)

        #count pixels of each component and sort them by size, excluding the background
        vol_list = []
        for label in label_unique:
            if label != 0:
                vol_list.append(np.count_nonzero(labels == label))

        #create binary array of only the largest component
        binary_mask = np.zeros(labels.shape)
        binary_mask = np.where(labels == vol_list.index(max(vol_list))+1, 1, 0)

        output_array[t,:,:] = binary_mask

    return output_array

def segmentation_yapic(im:ArrayLike,
                    temp_folder_path:str,
                    model_path:Path,
                    small_object_th:int=1000,
                    small_holes_th:int=200,
                    prediction_th:float=0.2,
                    show_debug:bool=True) -> ArrayLike:
    temp_folder_path = Path(temp_folder_path)
    if not os.path.isdir(temp_folder_path/'input'):
        os.mkdir(temp_folder_path/'input')
    if not os.path.isdir(temp_folder_path/'predict'):
        os.mkdir(temp_folder_path/'predict')

    mask = np.zeros(im.shape, dtype=bool)

    for i, im in enumerate(im):
        if show_debug:
            print(f'Writing temp tif {i}/{len(im)}')
        # create individual tif for each step
        imwrite(temp_folder_path/f'input/temp.tif', im)

        # predict the temp tif
        t = Session()
        t.load_prediction_data(str(temp_folder_path / f'input/temp.tif'),
                            str(temp_folder_path / f'predict/'))
        t.load_model(model_path)
        t.predict()

        # read the prediction and store it
        mask_t = np.squeeze(imread(str(temp_folder_path/'predict/temp_class_1.tif')))
        mask_t = mask_t > prediction_th
        mask_t = remove_small_objects(mask_t, small_object_th)
        mask_t = remove_small_holes(mask_t, small_holes_th)
        mask[i:i+1] = mask_t

    return mask

def segmentation_otsu(im:ArrayLike,
                    smoothingSigma:int=4,
                    clearMask:bool=True,
                    minRemoveSize:int=10000,
                    removeHoleSize:int=5000,
                    disk_footprint:int=1) -> ArrayLike:

    # Preallocating mask array
    footprint = morphology.footprints.disk(disk_footprint)
    mask = np.zeros(np.shape(im), dtype=bool)

    # Looping over the frames and
    for frame in np.arange(im.shape[0]):

        # Load each frame & automatically adjust contrast
        image = im[frame,...]
        image = exposure.equalize_hist(image)
        image =  exposure.rescale_intensity(image)

        # Use white_tophat and gaussian filter on the frame
        image = morphology.white_tophat(image, footprint)
        image = filters.gaussian(image, sigma = smoothingSigma)

        # Threshold the result & label each detected region
        thresholds = filters.threshold_multiotsu(image)
        regions = np.digitize(image, bins=thresholds)
        label_image = regions
        
        if clearMask:
        
            areas = []
            for region in regionprops(regions):
                areas.append(region.area)
            areas = np.array(areas)
            
            minRemoveSize = np.mean(areas)
            removeHoleSize = np.round(np.min(areas))

        # Fill the holes in the cell morphology and remove small objects/holes from the background
        fill = ndi.binary_fill_holes(label_image)

        mask[frame,...] = morphology.remove_small_holes(
            morphology.remove_small_objects(
                fill, min_size=minRemoveSize,
            removeHoleSize)

    return mask

def get_holes_mask(mask:ArrayLike) -> ArrayLike:
    inverted_mask= mask.copy() #copy the original mask
    # inverse the mask. The idea is to extract only the small object (= the wound) in the middle of the image, and to remove everything around.
    inverted_mask[mask==0]=1
    inverted_mask[mask==1]=0
    # remove objects touching the borders.
    mask_cleared=clear_border(inverted_mask)
    return mask_cleared