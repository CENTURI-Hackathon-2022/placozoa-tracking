import abc
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from placozoan.func_segmentation import (segmentation_chanvese, segmentation_otsu,
                               segmentation_yapic)
from placozoan.utils import package_installed


class SegmentationInterface(metaclass=abc.ABCMeta):

    def __init__(self, input:ArrayLike, params:Optional[Dict[str, Any]]) -> None:
        if params is None:
            params = {} # should load default
        self.params = params
        self.input = input

    @property
    @abc.abstractmethod
    def get_segmentation(self) -> ArrayLike:  # required
        """Compute the segmentation of the class"""
        raise NotImplementedError("Not implemented yet.")

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

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

    name = "Chan_Vese"

    def get_segmentation(self, input:Optional[ArrayLike]=None,
                    params:Optional[Dict[str, Any]]=None):
        input = np.array(input) if input else self.input
        params = params if params else self.params
        self.output = segmentation_chanvese(input, **params)
        return self.output

class YAPIC(SegmentationInterface):

    name = "YAPIC"

    def __init__(self, input: ArrayLike, params: Optional[Dict[str, Any]]) -> None:
        package_installed("tensorflow")
        package_installed("yapic")
        super().__init__(input, params)

    def get_segmentation(self, input:Optional[ArrayLike]=None,
                    params:Optional[Dict[str, Any]]=None):
        input = np.array(input) if input else self.input
        params = params if params else self.params
        self.output = segmentation_yapic(input, **params)
        return self.output

class Otsu(SegmentationInterface):

    name = "Otsu"

    def get_segmentation(self, input:Optional[ArrayLike]=None,
                    params:Optional[Dict[str, Any]]=None):
        input = np.array(input) if input else self.input
        params = params if params else self.params
        self.output = segmentation_otsu(input, **params)
        return self.output

