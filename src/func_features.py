import skimage
import napari
from tifffile import imread
from pathlib import Path
import scipy.ndimage as ndi

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table,perimeter
from skimage.transform import rotate
from skimage.morphology import convex_hull_image

from enum import Enum
from numpy.typing import ArrayLike
from functools import partial




def convexity(regionmask, intensity_image):
    """The convexity is the ratio between the perimeter
    of the convex hull of an object and the real perimeter
    of the object. If it is lower than 1, it indicates that
    the object is highly concave at some part of its shape.
    """
    conv_hull=convex_hull_image(regionmask) #compute the convex hull of the current image mask
    perimeter_convhull=perimeter(conv_hull) #compute its perimeter
    perimeter_mask=perimeter(regionmask) #compute the perimeter of the current image mask

    if perimeter_mask > 0:
        return perimeter_convhull/perimeter_mask #compute their ratio
    return 0

class ExtraProperties(Enum):
    convexity=partial(convexity)