import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import yaml
from einops import rearrange
from numpy.typing import ArrayLike
from tifffile import imread, imwrite


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
        a = np.array(rearrange(a, rearrange_pattern))
    if len(a.shape) != len(axes_out):
        raise ValueError('Missmatch between ', len(a.shape), len(axes_out),
        'the image should have the dimension like ', axes_out)
    imwrite(im_path, a, metadata={"axes": axes_out})
    return im_path

def package_installed(package_name:str) -> Literal[True]:
    # import specific dependency for deeplearning

    if package_name in sys.modules:
        return True
    elif (spec := importlib.util.find_spec(package_name)) is not None:
        return True
    else:
        raise ImportError(f"can't find the {package_name!r} module."
        f" You can install all the dependency using {'pip install .[all]'!r}")
