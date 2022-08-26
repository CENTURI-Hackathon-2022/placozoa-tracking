from enum import Enum
from functools import partial

from skimage.measure import perimeter
from skimage.morphology import convex_hull_image


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
