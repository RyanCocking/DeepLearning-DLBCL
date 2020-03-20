import tensorflow as tf

dir_img_data = "E:/OneDrive/Documents/UNIVERSITY/PhD/WellcomeTrust/rotation2/image_data"
dir_train = dir_img_data + "/train"
dir_test = dir_img_data + "/test"
dir_val = dir_img_data + "/val"
img_dim = 224    # Pixels
autotune = tf.data.experimental.AUTOTUNE

# Datasets
shuffle_buffer_size = 1000
test_split = 0.15    # Fraction of data to use in set
val_split = 0.15

# Training
load_model = True    # Load a model + classifier previously trained by the code
model_name = "MobileNetV2"    # Or VGG19
learning_rate = 1e-3
batch_size = 32    # No. images to use per gradient update
epochs = 10    # No. passes through entire dataset

file_suffix = "_lr{0:.1e}_bs{1:d}_ep{2:d}_{3:s}".format(learning_rate,
    batch_size, epochs, model_name)