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
test_mode = True   # If True, use a simpler neural net for easy testing
learning_rate = 1e-4
batch_size = 32    # No. images to use per gradient update
epochs = 10    # No. passes through entire dataset