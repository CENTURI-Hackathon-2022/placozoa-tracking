from typing import Any, Dict, List, Iterator, Optional, Sequence, Union, Tuple
from numpy.typing import ArrayLike
from pathlib import Path
import abc
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
import scipy.ndimage as ndi

from src.func_features import ExtraProperties
from src.func_features import convexity

class FeatureInterface(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def get_features(self) -> pd.DataFrame:  # required
        """Compute the segmentation of the class"""
        raise NotImplementedError("Not implemented yet.")

    @property
    def input_intensity(self) -> ArrayLike:
        return self._input_intensity

    @input_intensity.setter
    def input_intensity(self, val):
        self._input_intensity = val

    @property
    def input_mask(self) -> ArrayLike:
        return self._input_mask

    @input_mask.setter
    def input_mask(self, val):
        self._input_mask = val

    @property
    def output(self) -> pd.DataFrame:
        if hasattr(self, "_output"):
            return self._output
        self.get_features()
        return self._output

    @output.setter
    def output(self, val):
        self._output = val

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    @params.setter
    def params(self, val):
        self._params = val

class SingleObjectFeatures():

    def __init__(self,
            input_mask:ArrayLike,
            input_intensity:Optional[ArrayLike],
            params:Optional[Dict[str, Any]]) -> None:
        if params is None:
            params = {} # should load default
        self.params = params
        self.input_mask = input_mask
        self.input_intensity = input_intensity

    def get_features(self, input_mask:Optional[ArrayLike]=None,
                    input_intensity:Optional[ArrayLike]=None,
                    params:Optional[Dict[str, Any]]=None):

        # loading
        input_mask = np.array(input_mask) if input_mask else self.input_mask
        input_intensity = np.array(input_intensity) if input_intensity else self.input_intensity
        params = params if params else self.params
        extra_properties = tuple(ExtraProperties[prop_name].value.func for prop_name in params["extra_properties"])
        #preprocessing
        input_mask=ndi.binary_fill_holes(input_mask) # remove potential holes in the shape
        # extra_properties = (convexity,)
        ## Initialize the dataframe (final table) with timepoint 0.
        tab=pd.DataFrame(regionprops_table(
                input_mask[0].astype(int),
                intensity_image=input_intensity[0],
                properties=params["properties"],
                extra_properties=extra_properties,
            ))

        ## Then we will compute the metrics for each timepoint
        ## and concatenate the new line (corresponding to the new timepoint) to the dataframe
        for t in range (1,len(input_mask)) :
            other_timepoints=pd.DataFrame(regionprops_table(
                (input_mask[t]*(t+1)).astype(int),
                intensity_image=input_intensity[t],
                properties=params["properties"],
                extra_properties=extra_properties,
            ))
            tab=pd.concat([tab, other_timepoints])

        ppties=tab.set_index('label')
        self.output = ppties
        return self.output
