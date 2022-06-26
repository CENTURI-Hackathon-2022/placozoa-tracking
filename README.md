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
  
![IMG_0003](https://user-images.githubusercontent.com/15125196/175807730-910af1fc-75f0-4b17-9391-e628a45ed529.PNG)  

![IMG_0004](https://user-images.githubusercontent.com/15125196/175809001-1d9feea0-3564-40b7-ae37-ded15b6e761f.PNG)  


The following analysis have been developed: 
- **Shape segmentation (contouring the Trichoplax shape or its ablation wound shape):** 3 differents strategies have been investigated. They require the original movie as tif as input. They generate a tif file of the segmented shape (= binary mask).  
    1) A deep learning model has been trained on hand-made labelled bright-field trichoplax movie. The needed input file is a tif file with one channel correspond to the bright-field channel and one channel to the labels (Layer 1, the Trichoplax, Layer 2, the background). This kind of labelling can be done in Napari, before starting the training.
    2) Otsu's method: The code performs automatic thresholding to separate pixels into two classes of  foreground & background. This threshold is determined by by maximizing inter-class variance. A suitable candidate for creating shape masks of microscopic images of cells 
    3) Segmentation using morphsnakes: Finding outer edge of the animal using the morphsnakes algorithm for contour finding by comparing intensity averages inside and outside of a region.

![image](https://user-images.githubusercontent.com/94049435/175808887-5a489a01-8de8-4e63-aec6-fbbe07d8473a.png)



- **Preprocessing of the wound:** This code extract the wound shape from the Trichoplax mask (given as tif input) and generates a new tif mask only for the wound.
/!\ Running this part is necessary before doing the shape features analysis for the wound. 

- **Shape features analysis:** This code analyzes the object mask (either the Trichoplax or its wound) and compute shape features. Some of the features are computed from the region props Python module: you have to specify them in the 'props' list. Others can be added in the func_features.py, or new cell in the notebook. The output is a dataframe containing all the properties, saved as csv.  
/!\ This part needs to be run twice if you want to analyse the Trichoplax shape and its wound shape; **Don't forget to change mask filename input and dataframe name output ;)**

- **Plot generated shape data:** load the two previous csv files (Trichoplax and wound dataframes). It generates examples of analysis: area overtime, convexity, eccentricity, orientation angle, ... 

![image](https://user-images.githubusercontent.com/94049435/175808963-e0287fa3-3fd8-4975-8f00-da5cee2605d8.png)

- **Investigate flows within the Trichoplax: To understand the collective cell movements after laser ablation.**  
This code generates the velocity fields of pixels within the Trichoplax, from the global displacement of pixels between one timepoint to another. The wanted time window for the analysis (delta t), as well as the index of starting timepoint, the raw image name, folder for saving, should be specified as input. The output is a sequence of png images, that you can afterwhat load in FIJI and convert to gif file. This code could be improved by saving the coordinates of the vectors.  
![optical_flowoptical_flow_t50to_t55](https://user-images.githubusercontent.com/15125196/175806591-811e2830-d9a7-4d44-b405-787c8510210f.png)  

- **Segmentationn of Trichoplax lipid-containing cells**

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
