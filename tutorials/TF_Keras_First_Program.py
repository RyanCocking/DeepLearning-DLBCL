# My first TensorFlow program: https://www.tensorflow.org/tutorials/quickstart/beginner
# Keras sequential model: https://keras.io/getting-started/sequential-model-guide/

# Import throws warnings due to outdated TF version being used (1.5)
# as my laptop CPU does not support AVX drivers necessary for TF2.
import tensorflow as tf
from tensorflow import keras as tfk

# Import MNIST dataset (pre-processed images of handwritten numbers)
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert from ints to floats
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a sequential model by stacking layers (using Keras)
model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=(28, 28)),
    tfk.layers.Dense(128, activation='relu'),
    tfk.layers.Dropout(0.2),
    tfk.layers.Dense(10, activation='softmax')
])

# Choose an optimiser and loss (i.e. objective) function for training 
#
# (this also throws warnings, this time about 'keep_dims' being deprecated)
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# Train and evaluate model
print("==========================================================")
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

# The image classifier is now trained to ~ 98% accuracy on this dataset

# I still need to understand what is actually going on under the bonnet.
# What does 'accuracy' mean in the context of DL?