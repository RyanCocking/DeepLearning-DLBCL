# Deep learning
import os
import pathlib
import tensorflow as tf
import numpy as np
import dl_parameters as parm
from dl_routines import sort_images

path_obj = pathlib.Path(parm.dir_img_data)
class_names = ["ABC", "GCB"]

# Detect Windows and adjust file path to Linux version
if os.name == "nt":
    abc_paths = np.array([item.as_posix() for item in path_obj.glob("ABC/*.png")])
    gcb_paths = np.array([item.as_posix() for item in path_obj.glob("GCB/*.png")])
else:
    abc_paths = np.array([str(item) for item in path_obj.glob("ABC/*.png")])
    gcb_paths = np.array([str(item) for item in path_obj.glob("GCB/*.png")])


# If the training directory is empty or doesn't exist, begin to move image files
if os.path.exists(parm.dir_train) and os.path.isdir(parm.dir_train) and os.listdir(parm.dir_train):
    print("Training directory is not empty. Will not move image files.")
else:
    print("Moving ABC images to training, testing and validation folders...")
    sort_images(abc_paths, "ABC", parm.dir_train, parm.dir_test, parm.dir_val, parm.test_split, parm.val_split)
    print("Moving GCB images to training, testing and validation folders...")
    sort_images(gcb_paths, "GCB", parm.dir_train, parm.dir_test, parm.dir_val, parm.test_split, parm.val_split)
    print("Done") 


# The 1./255 is to convert from uint8 to float32 in range [0,1].
generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

# Generate batches of tensor image data, for all images
train_gen = generator.flow_from_directory(directory=parm.dir_train,
    batch_size=parm.batch_size, shuffle=True, 
    target_size=(parm.img_dim, parm.img_dim), classes=list(class_names))

test_gen = generator.flow_from_directory(directory=parm.dir_test,
    batch_size=parm.batch_size, shuffle=True, 
    target_size=(parm.img_dim, parm.img_dim), classes=list(class_names))

val_gen = generator.flow_from_directory(directory=parm.dir_val,
    batch_size=parm.batch_size, shuffle=True, 
    target_size=(parm.img_dim, parm.img_dim), classes=list(class_names))


    


