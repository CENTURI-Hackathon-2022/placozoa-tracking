from typing import List, Callable, Optional, Any, Dict
import numpy as np
import yaml
from pathlib import Path
from tifffile import imread, imwrite
from numpy.typing import ArrayLike

from einops import rearrange

def load_params(param_path:str) -> Optional[Dict[str, Any]]:
    params = None
    with open(param_path, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

def load_tiff(im_path:Path, axes_in:str="TZCYX",
            rearrange_pattern:Optional[str]=None) -> ArrayLike:
    raw_img = imread(im_path)
    if len(raw_img.shape) != len(axes_in):
        raise ValueError('Missmatch between ', len(raw_img.shape), len(axes_in),
        'the image should have the dimension like ', axes_in)
    if rearrange_pattern:
        raw_img = rearrange(raw_img, rearrange_pattern)
    return raw_img

def save_tiff(im_path:Path, a:ArrayLike, axes_out:str="TZCYX",
            rearrange_pattern:Optional[str]=None) -> Path:
    a = np.array(a)
    if rearrange_pattern:
        a = rearrange(a, rearrange_pattern)
    if len(a.shape) != len(axes_out):
        raise ValueError('Missmatch between ', len(a.shape), len(axes_out),
        'the image should have the dimension like ', axes_out)
    imwrite(im_path, a, metadata={"axes": axes_out})
    return im_path




class Repr:
    """Evaluable string representation of an object"""

    def __repr__(self): return f'{self.__class__.__name__}: {self.__dict__}'

class Wrapper(Repr):
    """A function wrapper that returns a partial for input only."""

    def __init__(self, function: Callable, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)

    def __call__(self, inp: np.ndarray): return self.function(inp)

class Compose:
    """Baseclass - composes several transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __repr__(self): return str([transform for transform in self.transforms])