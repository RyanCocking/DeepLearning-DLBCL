dir_img_data = "E:/OneDrive/Documents/UNIVERSITY/PhD/WellcomeTrust/rotation2/image_data"
dir_train = dir_img_data + "/train"
dir_test = dir_img_data + "/test"
dir_val = dir_img_data + "/val"
img_dim = 224    # Pixels

# Datasets
test_split = 0.15         # Fraction of data to use in set. NOTE: changing will
val_split = test_split    # require images to be moved to other directories

# Training
load_model = False    # Load a model + classifier previously trained by the code
model_name = "MobileNetV2"    # MobielNetV2 or VGG19
learning_rate = 3e-5
batch_size = 64    # No. images to use per gradient update
epochs = 200   # No. passes through entire dataset

file_suffix = "_lr{0:.1e}_bs{1:d}_sp{2:.2f}_{3:s}".format(learning_rate,
    batch_size, test_split, model_name)

# 256 gives oom