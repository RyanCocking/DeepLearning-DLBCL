# Deep learning
import os
import sys
import pathlib
import tensorflow as tf
import numpy as np
import dl_parameters as parm
from dl_routines import sort_images
import matplotlib.pyplot as plt
import pickle
import time

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

# Construct model from pre-trained ConvNet
if not parm.load_model:
    # Load the pre-trained model without its topmost classification layer
    print("Loading pre-trained model...")
    if parm.model_name == "MobileNetV2":
        pre_model = tf.keras.applications.MobileNetV2(input_shape=img_shape, 
                                                      include_top=False, 
                                                      weights='imagenet')
    elif parm.model_name == "VGG19":
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
    
    
    # Stack these layers on top of the VGG model using a Keras sequential model
    print("Stacking classification layers on top of pre-trained model...")
    model = tf.keras.Sequential(layers=[pre_model, global_average_layer, prediction_layer])
        
    # Compile the model. Use a binary cross-entropy loss function, since there are
    # only two classes
    print("Compiling combined Sequential model...")
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=parm.learning_rate), 
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                  metrics=['accuracy'])
    
    # Training and validation
    print("Training model...")
    start = time.time()
    train_results = model.fit(x=train_gen, epochs=parm.epochs, validation_data=val_gen)
    end = time.time()
    t = end - start
    print("Training time = {0:.2f} s".format(t))
    
    print("Saving training results and model state...")
    model.save("Model{0}.h5".format(parm.file_suffix))
    with open("TrainingResults{0}.pkl".format(parm.file_suffix), 'wb') as f:
        pickle.dump(train_results.history, f, pickle.HIGHEST_PROTOCOL)
    
# Load a model previously trained by the code
elif parm.load_model:
    print("Loading previously trained classifier...")
    model = tf.keras.models.load_model("Model{0}.h5".format(parm.file_suffix))
else:
    print("ERROR - No model loaded in for training. Exiting program.")
    sys.exit()

# Testing the trained model
print("Testing model...")
start = time.time()
test_results = model.evaluate(x=test_gen, verbose=1)
end = time.time()
t = end - start
print("Testing time = {0:.2f} s".format(t))

print("Saving testing results...")
np.savetxt("TestingResults{0}.txt".format(parm.file_suffix), test_results,
           header="Loss, accuracy")

print("Making predictions...")
start = time.time()
predictions = model.predict(x=test_gen, steps=100, verbose=1)
end = time.time()
t = end - start
print(predictions.shape)
print(predictions[0])
print(predictions[1])
print("Prediction time = {0:.2f} s".format(t))
print("Saving predictions...")
np.savetxt("Predictions{0}.txt".format(parm.file_suffix), predictions,
           header = "{0} class predictions".format(predictions.shape[0]))
    
if not parm.load_model:
    # Plotting
    print("Plotting results...")
    acc = train_results.history['accuracy']
    val_acc = train_results.history['val_accuracy']
    loss = train_results.history['loss']
    val_loss = train_results.history['val_loss']
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1)
    plt.plot(acc, 'ro', label='Tr acc')
    plt.plot(val_acc, 'bo', label='Val acc')
    plt.ylim([0,1.0])
    plt.title('Accuracy and loss')
    plt.plot(loss, 'rD', label='Tr loss')
    plt.plot(val_loss, 'bD', label='Val loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.savefig("AccuracyLoss{0}.png".format(parm.file_suffix), dpi=300)
    
print("Done")
    



    


