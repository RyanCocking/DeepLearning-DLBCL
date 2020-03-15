# Routines for deep learning
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
    make_dir(dir_train)
    make_dir(dir_test)
    make_dir(dir_val)
        
    # Move the images to their respective directories
    move_images(train_paths, dir_train_class)
    move_images(test_paths, dir_test_class)
    move_images(val_paths, dir_val_class)
    