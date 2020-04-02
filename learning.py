# Deep learning
import os
import sys
import pathlib
import tensorflow as tf
import numpy as np
import dl_parameters as parm
from dl_routines import sort_images, construct_model, ask_for_overwrite
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
# augmentation in the form of random horizontal and vertical flips to training
# data only. Augmentation is done on-the-fly as the model is trained with the
# generator.
aug_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255, 
                                                          horizontal_flip=True, 
                                                          vertical_flip=True)
# Do not augment validation or test data
gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

# Generate batches of tensor image data, for all images
print("Loading data as shuffled batches...")
train_gen = aug_gen.flow_from_directory(directory=parm.dir_train, 
                                          batch_size=parm.batch_size, 
                                          shuffle=True, 
                                          target_size=(parm.img_dim, parm.img_dim), 
                                          classes=list(class_names), 
                                          class_mode='binary')

test_gen = gen.flow_from_directory(directory=parm.dir_test, 
                                         batch_size=parm.batch_size, 
                                         shuffle=True, 
                                         target_size=(parm.img_dim, parm.img_dim), 
                                         classes=list(class_names), 
                                         class_mode='binary')

val_gen = gen.flow_from_directory(directory=parm.dir_val, 
                                        batch_size=parm.batch_size, 
                                        shuffle=True, 
                                        target_size=(parm.img_dim, parm.img_dim), 
                                        classes=list(class_names), 
                                        class_mode='binary')

img_shape = (parm.img_dim, parm.img_dim, 3)

# Construct model from pre-trained ConvNet
times = []
if not parm.load_model:
    
    print("Constructing classification model from pre-trained ConvNet...")
    model = construct_model(parm.model_name, img_shape, parm.learning_rate)
    
    # Training and validation
    print("Training model...")
    start = time.time()
    train_results = model.fit(x=train_gen, epochs=parm.epochs,
                              validation_data=val_gen)
    end = time.time()
    t = end - start
    times.append(t)
    print("Training time = {0:.2f} s".format(t))
    
    # Ask permission to overwrite if file exists. This permission will apply to
    # all other files created by the code. If file doesn't exist, write anyway.
    print("Saving mo del state...")
    fname = "Model{0}.h5".format(parm.file_suffix)
    allow_ow = ask_for_overwrite(fname, force_overwrite=True)
    if allow_ow:
        model.save(fname)
    
    print("Saving training results...")
    if allow_ow:
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
times.append(t)
print("Testing time = {0:.2f} s".format(t))

print("Saving testing results...")
if allow_ow:
    np.savetxt("TestingResults{0}.txt".format(parm.file_suffix),
               test_results, header="Loss, accuracy")

print("Making predictions...")
start = time.time()
predictions = model.predict(x=test_gen, steps=100, verbose=1)
end = time.time()
t = end - start
times.append(t)
print("Made {0:d} predictions".format(predictions.shape[0]))
print("Prediction time = {0:.2f} s".format(t))

print("Saving predictions...")
if allow_ow:
    np.savetxt("Predictions{0}.txt".format(parm.file_suffix), predictions,
               header = "{0} class predictions".format(predictions.shape[0]))
 
print("Saving times...")
if allow_ow:
    np.savetxt("Times{0}.txt".format(parm.file_suffix), times,
               header = "Training (unless pre-loaded), testing and prediction times (s)")

# Plotting
print("Plotting results...") 
if not parm.load_model:   
    acc = train_results.history['accuracy']
    val_acc = train_results.history['val_accuracy']
    loss = train_results.history['loss']
    val_loss = train_results.history['val_loss']    
elif parm.load_model:
    history = pickle.load(open(
            "TrainingResults{0}.pkl".format(parm.file_suffix), 'rb'))
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    
plt.figure(figsize=(8, 8))
plt.subplot(1, 1, 1)
plt.plot(acc, 'r-', label='Train acc')
plt.plot(val_acc, 'r--', label='Val acc')
plt.ylim([0.4,1.0])
plt.xlim([0, parm.epochs+1])
plt.title('Accuracy and loss')
plt.plot(loss, 'b-', label='Train loss')
plt.plot(val_loss, 'b--', label='Val loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
if allow_ow:
    plt.savefig("AccuracyLoss{0}.png".format(parm.file_suffix), dpi=300)

print("Done")
    



    


