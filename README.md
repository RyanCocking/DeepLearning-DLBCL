DeepLearning-DLBCL
---
Use a convolutional neural network, [MobileNetV2](https://arxiv.org/abs/1801.04381), to identify the activated B-cell-like (ABC) and germinal centre B-cell-like (GCB) classes of diffuse large B-cell lymphoma (DLBCL) from immunohistochemistry-stained histopathology slides. 

The software currently gives a classification accuracy of around 67% and overfits to the training set, but I think this may be an issue with the image preprocessing.

Developed during a first-year PhD rotation project, supervised by [Professor David R Westhead](https://biologicalsciences.leeds.ac.uk/molecular-and-cellular-biology/staff/154/professor-david-r-westhead) at the Leeds Institute for Data Analytics, University of Leeds.

Requirements
---
* Linux

**apt-get:**
* python >= 3.6.9
* openslide-tools

**pip-install:**
* openslide-python
* opencv-python
* pandas
* xlrd
* numpy
* tensorflow-gpu >= 2.1.0

## Running
* Set directories and constants: `python parameters.py` and `python dl_parameters.py`
* Generate images: `python main.py <gene_name>` where `gene_name` is `ABC` or `GCB`
* Populate datasets and train a model: `python learning.py`
