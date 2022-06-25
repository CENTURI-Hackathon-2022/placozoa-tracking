import numpy as np
from scipy import ndimage as ndi
from skimage import (filters, morphology, exposure)

from tifffile import imread, imwrite
from skimage import filters
from pathlib import Path



def shapeExtraction(filePath, smoothingSigma = 4, minRemoveSize = 10000, removeHoleSize = 5000):
    
    # Define path to the .tiff file and reading the image
    path = Path(filePath)
    im = imread(path)


    # Preallocating mask array
    footprint = morphology.footprints.disk(1)
    mask = np.zeros(np.shape(im))

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

        # Fill the holes in the cell morphology and remove small objects/holes from the background
        fill = ndi.binary_fill_holes(label_image)

        mask[frame,...] = morphology.remove_small_holes(
            morphology.remove_small_objects(
                fill, min_size=minRemoveSize),
            removeHoleSize)
            
     
    
    # Save the resulting mask
    imwrite('mask.tif', mask, photometric='minisblack')

    return mask