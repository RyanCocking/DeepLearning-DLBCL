# Routines for deep learning
import sys
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

def check_devices():
    """
    Brief check to see if the system has a CUDA-capable GPU. Halts the program
    if none are found.
    """

    print("==================================================================")
    print("=================== CHECKING FOR LOCAL DEVICES ===================")
    print("==================================================================")
    print("All local devices:")
    print("")
    print(device_lib.list_local_devices())        # List of CPUs and GPUs
    x = tf.config.list_physical_devices('GPU')    # List of GPUs
    y = tf.test.is_built_with_cuda()
    print("")
    print("GPU devices:")
    print("")
    print(x)
    print("")
    print("TensorFlow built with CUDA = {0}".format(y))
    print("")
    if x is not None and y is True:
        print("CUDA-capable GPU found. System has OK setup for deep learning")
    else:
        print("WARNING: No CUDA-capable GPU found. Deep learning will be\n"
              "         significantly slower")
        print("")
        print("Program exited")
        quit()
    
    print("==================================================================")
    print("========================= CHECK COMPLETE =========================")
    print("==================================================================")

def ask_for_overwrite(fname, force_overwrite):
    if force_overwrite:
        return True
    # If path exists, ask user for overwrite permission
    if os.path.exists(fname):
        print("File: {0} already exists. Overwrite? (y/n)".format(fname))
        while True:
            ow = input()
            if ow != "y" and ow != "n":
                print("Please type either 'y' or 'n'.")
            elif ow == "y":
                print("Writing file {0}...".format(fname))
                return True
            else:
                print("File not written")
                return False
    # OK to write file if path does not exist
    elif not os.path.exists(fname):
        return True

def make_dir(path):
    try:
        # Will create all required intermediate folders
        os.makedirs(path)
    except FileExistsError:
        pass

def move_images(paths, target_dir):
    for old_path in paths:
        img_file = old_path.split("/")[-1]  
        shutil.move(old_path, "{0}/{1}".format(target_dir, img_file))

def sort_images(img_paths, class_name, dir_train, dir_test, dir_val, test_split, 
    val_split):
    """
    Randomly shuffle images into folders suitable for the Keras API to create
    training, testing and validation sets.
    
    Our image generator code sorts images into img_dir/class_name, but we need
    to create a structure like img_dir/set_name/class_name.
    
    arguments:
        img_paths: numpy str array, paths to image files for a single class
        class_name: str, the name of a single class
        img_dir: str, the master directory containing the class directories
        test_split, val_split: float, the fractional splits of the data into
                               testing and validation sets
    """
    
    # Randomise the order of the image paths
    num_img = img_paths.size
    random_paths = np.random.permutation(img_paths)
    
    # Number of images per set
    test_size = int(test_split * num_img)
    val_size = int(val_split * num_img)
    train_size = num_img - test_size - val_size
    
    # Take slices of the randomised array
    train_paths = random_paths[:train_size] 
    test_paths = random_paths[train_size:train_size+test_size]
    val_paths = random_paths[train_size+test_size:]
    
    # Create the directories
    dir_train_class = "{0}/{1}".format(dir_train, class_name)
    dir_test_class = "{0}/{1}".format(dir_test, class_name)
    dir_val_class = "{0}/{1}".format(dir_val, class_name)
    make_dir(dir_train_class)
    make_dir(dir_test_class)
    make_dir(dir_val_class)
        
    # Move the images to their respective directories
    move_images(train_paths, dir_train_class)
    move_images(test_paths, dir_test_class)
    move_images(val_paths, dir_val_class)
    
def construct_model(model_name, img_shape, learning_rate):
    """
    Load a pre-trained convolutional neural network, freeze its trainable layers
    and add a classifier on top (consisting of an averaging layer and dense
    layer).
    
    Compiled with a Stochastic Gradient Descent optimiser and Binary Cross-Entropy
    loss function.
    
    arguments:
        model_name: str, the name of the pre-trained ConvNet. So far accepts
                    MobileNetV2 and VGG19.
        img_shape: int, the dimensions of the square images that will be passed
                   to the modek for training. Generalisable to rectangles.
        learning_rate: float, a hyperparameter that controls the rate of
                       gradient updates whilst training the model.
    returns:
        model: a compiled Keras model, ready to be assigned to training data.
    """
    
    # Load the pre-trained model without its topmost classification layer
    if model_name == "MobileNetV2":
        pre_model = tf.keras.applications.MobileNetV2(input_shape=img_shape, 
                                                      include_top=False, 
                                                      weights='imagenet')
    elif model_name == "VGG19":
        pre_model = tf.keras.applications.vgg19.VGG19(input_shape=img_shape, 
                                                      include_top=False, 
                                                      weights='imagenet')
    else:
        print("ERROR - Invalid model name. Exiting program.")
        sys.exit()
    
    # Freeze the convolutional base of the model, preventing the weights from being
    # updated during training. 
    #
    # This is especially important given that the base model has many millions of
    # layers, and that we are more interested in using these layers to do some
    # classification rather than train them further.
    pre_model.trainable = False
    
    # Add a classifier to the top of the model. The pooling layer converts features
    # to a 1280-element vector per image, whilst the dense layer converts these
    # features into a single prediction per image.
    #
    # These are the layers that we want to train for image classification.
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)
    
    pre_model.summary()
    
    # Stack these layers on top of the VGG model using a Keras sequential model
    model = tf.keras.Sequential(layers=[pre_model, global_average_layer, 
                                        prediction_layer])
        
    # Compile the model. Use a binary cross-entropy loss function, since there are
    # only two classes, and a Stochastic Gradient Descent optimiser.
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=learning_rate), 
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                  metrics=['accuracy'])
    
    return model
    