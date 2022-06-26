from tifffile import imread, imwrite
import scipy.ndimage as nd
from pathlib import Path
from skimage.morphology import disk
import numpy as np
import morphsnakes as ms
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import rank
from yapic.session import Session

from typing import Any, Dict, List, Iterator, Optional, Sequence, Union, Callable
from numpy.typing import ArrayLike

import abc
from src.utils import load_tiff, save_tiff
import os


class Repr:
    '''Evaluable string representation of an object'''

    def __repr__(self): return f'{self.__class__.__name__}: {self.__dict__}'


class FunctionWrapperSingle(Repr):
    '''A function wrapper that returns a partial for input only.'''

    def __init__(self, function: Callable, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)

    def __call__(self, inp: np.ndarray): return self.function(inp)

class Compose:
    '''Baseclass - composes several transforms together.'''

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __repr__(self): return str([transform for transform in self.transforms])

# transforms = ComposeDouble([
#     FunctionWrapperDouble(create_dense_target, input=False, target=True),
#     FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
#     FunctionWrapperDouble(normalize_01)
# ])




class ComposeDouble(Compose):
    '''Composes transforms for input-target pairs.'''

    def __call__(self, inp: np.ndarray, target: dict):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target


def create_dense_target(tar: np.ndarray):
    classes = np.unique(tar)
    dummy = np.zeros_like(tar)
    for idx, value in enumerate(classes):
        mask = np.where(tar == value)
        dummy[mask] = idx

    return dummy


class Method():

    @classmethod
    @property
    @abc.abstractmethod
    def segment(self):  # required
        '''Assign acroym for a subclass dataset'''
        ...

    @property
    @abc.abstractmethod
    def input_paths(self):  # required
        '''Path to input files'''
        ...

    @property
    @abc.abstractmethod
    def output_paths(self):  # required
        '''Path to output files'''
        ...

class SegmentationMethod(Method):

    @property
    def output_paths(self):
        return



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