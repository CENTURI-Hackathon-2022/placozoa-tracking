from typing import Any, Dict, List, Iterator, Optional, Sequence, Union, Tuple
from numpy.typing import ArrayLike
from pathlib import Path
import abc
import numpy as np

from src.func_segmentation import segmentation_chanvese, segmentation_yapic

class SegmentationInterface(metaclass=abc.ABCMeta):

    def __init__(self, input:ArrayLike, params:Optional[Dict[str, Any]]) -> None:
        if params is None:
            params = {} # should load default
        self.params = params
        self.input = input

    @property
    @abc.abstractmethod
    def get_segmentation(self):  # required
        """Compute the segmentation of the class"""
        raise NotImplementedError("Not implemented yet.")

    @property
    def input(self) -> ArrayLike:
        return self._input

    @input.setter
    def input(self, val):
        self._input = val

    @property
    def output(self) -> ArrayLike:
        if hasattr(self, "_output"):
            return self._output
        self.get_segmentation()
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

class ChanVese(SegmentationInterface):

    def get_segmentation(self, input:Optional[ArrayLike]=None,
                    params:Optional[Dict[str, Any]]=None):
        input = np.array(input) if input else self.input
        params = params if params else self.params
        self.output = segmentation_chanvese(input, **params)
        return self.output

class YAPIC(SegmentationInterface):

    def get_segmentation(self, input:Optional[ArrayLike]=None,
                    params:Optional[Dict[str, Any]]=None):
        input = np.array(input) if input else self.input
        params = params if params else self.params
        self.output = segmentation_yapic(input, **params)
        return self.output
