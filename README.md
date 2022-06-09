# Placozoa tracking

This project is about a [placozoa] that has been cut in its center.

# Getting started

## Installing the dependencies
To install Python and the required dependencies we strongly recommend to use
[conda], [mamba] or [pipenv].

## Installing conda

Conda can be installed multiple ways. There is no recommendations about how to
but one can read [there](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
for a likely exhaustive list on ways to install conda.

Note that Anaconda is not necessarily recommended, [miniconda] might be a better
alternative.

Moreover, it is advised to start jupyter notebooks from a shell/terminal/prompt
to be able to better see the error messages.

## Installing the dependencies

Once conda is installed (or your favorite environment manager), you can create
and activate your environment:
```shell
conda create -n placozoa
conda activate placozoa
```

Then, there is a `setup.py` file with the basic dependencies present within this
repository. It means that you can easily install all the likely necessary
dependencies using [pip]. It might be necessary to install it first:
```shell
conda install pip
```

Then, it is possible to install the dependencies, from the placozoa-tracking
folder the following way:
```shell
pip install .
```

### List of dependencies:
Here is the list of dependencies that will be installed:
- [numpy] : basic library for array manipulation
- [matplotlib] : basic library to plot figures
- [scipy] : one of the basic libraries for image analysis (`ndimage`)
- [scikit-image] : one of the basic libraries for image analysis
- [scikit-learn] : one of the basic libraries for data analysis
- [tifffile] : library to read and write tiff images
- [ipython] : interactive python terminal
- [jupyter] : python notebook
- [napari] : 3D image visualizer

# Road Map

# Problems that this software is trying to solve
Understanding better Placozoans wound healing

# Milestones
The milestones are likely in order of difficulty but they can be done in any.

It is not expected, whatsoever, for all the milestones to be completed.

- Detecting and tracking the placozoan throughout the whole movie
- Quantitatively characterising the shape of the placozoan over time
- Detecting and tracking the wound throughout the whole movie
- Quantitatively characterising the shape of the wound over time
- Building a napari plugin for the detection and tracking
- Computing and quantifying the cell flows within the placozoan
- Putting in relation placozoan and wound shape with cell flows

You can find the roadmap for the project as an issue [there](https://github.com/CENTURI-Hackathon-2022/placozoan-visualisation/issues/1).

You should try to follow the milestones but the order is not important (though
some milestones are dependent on others).


[conda]: https://docs.conda.io/en/latest/
[mamba]: https://mamba.readthedocs.io/en/latest/
[pipenv]: https://pipenv.pypa.io/en/latest/
[miniconda]: https://docs.conda.io/en/latest/miniconda.html
[placozoa]: https://en.wikipedia.org/wiki/Placozo
[pip]: https://pypi.org/project/pip
[numpy]: https://numpy.org
[scipy]: https://scipy.org
[matplotlib]: https://matplotlib.org
[scikit-image]: https://scikit-image.org
[scikit-learn]: https://scikit-learn.org
[tifffile]: https://pypi.org/project/tifffile
[ipython]: https://ipython.org
[jupyter]: https://jupyter.org
[napari]: https://napari.org
