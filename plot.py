# Plot stuff
import pickle
import matplotlib.pyplot as plt
import glob

train_result_paths = glob.glob("Training*.pkl")
colours = ["r", "b", "g", "m", "c", "k", "y"]
lines = ["-", "--", ":"]

plt.figure(figsize=(8, 8))
plt.subplot(1, 1, 1)
plt.ylim([0.4,0.8])
plt.xlim([-1, 61])
plt.ylabel("Loss")
plt.xlabel('Epoch')
for i, path in enumerate(train_result_paths):
    history = pickle.load(open(path, "rb"))
    loss = history['loss']
    val_loss = history['val_loss']
    split = path.split("_")
    plt.plot(loss, ls=lines[0], c=colours[i], label="TRA {0:s}, {1:s}, {2:s}".format(split[1], split[2], split[3]))
    plt.plot(val_loss, ls=lines[1], c=colours[i], label="VAL {0:s}, {1:s}, {2:s}".format(split[1], split[2], split[3]))

plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig("Loss.png", dpi=300)

plt.figure(figsize=(8, 8))
plt.subplot(1, 1, 1)
plt.ylim([0.4,0.8])
plt.xlim([-1, 61])
plt.ylabel("Accuracy")
plt.xlabel('Epoch')
for i, path in enumerate(train_result_paths):
    history = pickle.load(open(path, "rb"))
    acc = history["accuracy"]
    val_acc = history['val_accuracy']
    split = path.split("_")
    plt.plot(acc, ls=lines[0], c=colours[i], label="TRA {0:s}, {1:s}, {2:s}".format(split[1], split[2], split[3]))
    plt.plot(val_acc, ls=lines[1], c=colours[i], label="VAL {0:s}, {1:s}, {2:s}".format(split[1], split[2], split[3]))

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("Accuracy.png", dpi=300)