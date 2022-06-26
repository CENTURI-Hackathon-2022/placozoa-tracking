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

# Objectives
The objectives are likely in order of difficulty but they can be done in any.

It is not expected, whatsoever, for all the objectives to be completed.

- Detecting and tracking the placozoan throughout the whole movie #2
- Quantitatively characterising the shape of the placozoan over time #3
- Detecting and tracking the wound throughout the whole movie #4
- Quantitatively characterising the shape of the wound over time #5
- Building a napari plugin for the detection and tracking #6
- Computing and quantifying the cell flows within the placozoan #7
- Putting in relation placozoan and wound shape with cell flows #8

You can find the roadmap for the project as an issue [there](https://github.com/CENTURI-Hackathon-2022/placozoan-visualisation/issues/1).

You should try to follow the objectives but the order is not important (though
some milestones are dependent on others).

## Objective dependencies
&rarr; #2 

&rarr; #4 

#2 &rarr; #3 

#4 &rarr; #5

#2 | #4 &rarr; #6

&rarr; #7 

(#3 | #4) & #7 &rarr; #8

Legend:
- #x &rarr; #y: #x needs to be completed before #y can be started
- #x | #y: #x __or__ #y needs to be completed
- #x & #y: #x __and__ #y needs to be completed

# Tutorial
The following analysis have been developed: 
- Shape segmentation (contouring the Trichoplax shape or its ablation wound shape): 3 differents strategies have been investigated.  
1) A deep learning model has been trained on labelled bright-field trichoplax movie. It generates a tif file of the segmented shape (= binary mask).  
2)  
3)  

- Preprocessing of the wound: This code extract the wound shape from the Trichoplax mask (given as input) and generates a new tif mask only for the wound.
/!\ Running this part is necessary before doing the shape features analysis for the wound. 

- Shape features analysis: This code analyzes the object mask (either the Trichoplax or its wound) and compute shape features. Some of the features are computed from the region props Python module: you have to specify them in the 'props' list. Others can be added in the func_features.py, or new cell in the notebook. The output is a dataframe containing all the properties, saved as csv.  
/!\ This part needs to be run twice if you want to analyse the Trichoplax shape and its wound shape; **Don't forget to change mask filename input and dataframe name output ;)**

- Plot generated shape data: load the two previous csv files (Trichoplax and wound dataframes). It generates examples of analysis: area overtime, convexity, eccentricity, orientation angle, ... 

- Investigate flows within the Trichoplax: This code generates the velocity fields of pixels within the Trichoplax, from the global displacement of pixels between one timepoint to another. The wanted time windows for the analysis (delta t), as well as raw image name, folder for saving, should be specified as input.
![optical_flowoptical_flow_t50to_t55](https://user-images.githubusercontent.com/15125196/175806591-811e2830-d9a7-4d44-b405-787c8510210f.png)

- Tracking of Trichoplax lipid-containing cells

[conda]: https://docs.conda.io/en/latest/
[mamba]: https://mamba.readthedocs.io/en/latest/
[pipenv]: https://pipenv.pypa.io/en/latest/
[miniconda]: https://docs.conda.io/en/latest/miniconda.html
[placozoa]: https://en.wikipedia.org/wiki/Placozoa
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
