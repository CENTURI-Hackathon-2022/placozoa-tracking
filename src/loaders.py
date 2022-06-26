import yaml
from typing import Any, Dict, List, Iterator, Optional, Sequence, Union, Tuple

class ParameterLoader():

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @parameters.setter
    def parameters(self, val):
        self._parameters = val

    def __init__(self, param_path) -> None:
        # load parameters from yaml file
        with open("example.yaml", "r") as stream:
            try:
                self.parameters = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

class ImageLoader():
