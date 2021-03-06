# DeepLearning-DLBCL
Using a convolutional neural network to identify the activated B-cell-like (ABC) and germinal centre B-cell-like (GCB) classes of diffuse large B-cell lymphoma (DLBCL) IHC-stained histopathology slide images. 

The network (MobileNetV2) currently gives a classification accuracy of around 67% and overfits to the training set, but I think this may be an issue with the image preprocessing.

Code initially developed during a first-year PhD rotation project funded by the Wellcome Trust, supervised by Professor David R Westhead at the Leeds Institute for Data Analytics, University of Leeds.

## Requirements

python >= 3.6.9

apt-get install: openslide-tools

pip install: openslide-python, opencv-python, pandas, xlrd, numpy, tensorflow-gpu (>= 2.1.0)

tensorflow-gpu requires the appropriate versions of CUDA and cuDNN depending on your system. 

The preprocessing code was developed and tested in a Linux subsystem and the learning code in an Anaconda environment (Spyder), both in Windows 10.

## Running

Set directories and constants in `parameters.py` and `dl_parameters.py` (soon to be merged)

To generate images, run `python main.py gene_name` with either ABC or GCB as the argument

To populate datasets and train a model, run `python learning.py`
