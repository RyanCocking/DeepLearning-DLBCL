# DeepLearning-DLBCL
Using a convolutional neural network to identify the activated B-cell-like (ABC) and germinal centre B-cell-like (GCB) classes of diffuse large B-cell lymphoma (DLBCL) IHC-stained histopathology slide images. 

The network (MobileNetV2) currently gives poor classification accuracy of around 67% and overfits to the training set, but I think this may be an issue with the image preprocessing.

Code initially developed during a first-year PhD rotation project funded by the Wellcome Trust, supervised by Professor David R Westhead at the Leeds Institute for Data Analytics, University of Leeds.

REQUIREMENTS

python >= 3.6.9

apt-get install: openslide-tools

pip install: openslide-python, opencv-python, pandas, xlrd, numpy, tensorflow-gpu (>= 2.1.0)

tensorflow-gpu requires the appropriate versions of CUDA and cuDNN depending on your system. 

The preprocessing code was developed and tested in a Linux subsystem and the learning code in an Anaconda environment (Spyder), both in Windows 10.
