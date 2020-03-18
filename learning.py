# Deep learning
import os
import pathlib
import tensorflow as tf
import numpy as np
import dl_parameters as parm
from dl_routines import sort_images
from tensorflow.keras.preprocessing.image import img_to_array, load_img


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
    sort_images(abc_paths, "ABC", parm.dir_train, parm.dir_test, parm.dir_val, 
                parm.test_split, parm.val_split)
    print("Moving GCB images to training, testing and validation folders...")
    sort_images(gcb_paths, "GCB", parm.dir_train, parm.dir_test, parm.dir_val, 
                parm.test_split, parm.val_split)
    print("Done") 


# The 1./255 is to convert from uint8 to float32 in range [0,1]. Apply data
# augmentation in the form of random horizontal and vertical flips.
generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255, 
                                                            horizontal_flip=True, 
                                                            vertical_flip=True)

# Generate batches of tensor image data, for all images
print("Loading data as shuffled, augmented batches...")
train_gen = generator.flow_from_directory(directory=parm.dir_train, 
                                          batch_size=parm.batch_size, 
                                          shuffle=True, 
                                          target_size=(parm.img_dim, parm.img_dim), 
                                          classes=list(class_names), 
                                          class_mode='binary')

test_gen = generator.flow_from_directory(directory=parm.dir_test, 
                                         batch_size=parm.batch_size, 
                                         shuffle=True, 
                                         target_size=(parm.img_dim, parm.img_dim), 
                                         classes=list(class_names), 
                                         class_mode='binary')

val_gen = generator.flow_from_directory(directory=parm.dir_val, 
                                        batch_size=parm.batch_size, 
                                        shuffle=True, 
                                        target_size=(parm.img_dim, parm.img_dim), 
                                        classes=list(class_names), 
                                        class_mode='binary')

img_shape = (parm.img_dim, parm.img_dim, 3)

# Load the pre-trained model without the topmost classification (or 'bottleneck)
# layers (include_top=False)
print("Loading pre-trained model...")
if parm.test_mode is True:
    pre_model = tf.keras.applications.MobileNetV2(input_shape=img_shape, 
                                              include_top=False, 
                                              weights='imagenet')
else:
    pre_model = tf.keras.applications.vgg19.VGG19(input_shape=img_shape, 
                                                  include_top=False, 
                                                  weights='imagenet')

pre_model.summary()

# Freeze the convolutional base of the model prior to compilation and training.
# This prevents weights from being updated during training.
pre_model.trainable = False

# Add a classifier to the top of the model. The pooling layer converts features
# to a 1280-element vector per image, whilst the dense layer converts these
# features into a single prediction per image.
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)

# Stack these layers on top of the VGG model using a Keras sequential model
print("Stacking classification layers on top of pre-trained model...")
model = tf.keras.Sequential(layers=[pre_model, global_average_layer, prediction_layer])
    
# Compile the model. Use a binary cross-entropy loss function, since there are
# only two classes
print("Compiling combined Sequential model...")
model.compile(optimizer=tf.keras.optimizers.SGD(lr=parm.learning_rate), 
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
              metrics=['accuracy'])
model.summary()

print("Training model...")
history = model.fit(x=train_gen, epochs=parm.epochs, validation_data=val_gen)

print("Done")




    


