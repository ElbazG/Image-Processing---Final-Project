import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from distutils.dir_util import copy_tree, remove_tree
from random import randint
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D

"""

Hyper-Parameters:
1. IMG_SIZE - Size of images to the CNN
2. DIRECTORY - The directory that stores all our Dataset images
3. CLASS - Our classifications options for each image
4. ROOT, TEST, TRAIN and WORK - path to relevant files
5. DIM - Dimension of our images

"""
DIRECTORY = "C:/Users/Gil/PycharmProjects/img_process/Union"
ROOT = "./"
TEST = DIRECTORY + "/Test"
TRAIN = DIRECTORY + "/Train"
WORK = ROOT + "/Dataset"

"""
We check if there is already a directory named like this and if it does then remove it
and create a new clean directory named Dataset
"""
if os.path.exists(WORK):
    remove_tree(WORK)

os.mkdir(WORK)
copy_tree(TRAIN, WORK)
copy_tree(TEST, WORK)
print("Directory includes:", os.listdir(WORK))

WORK_DIR = './dataset/'

CLASS = ['NonDemented',
           'VeryMildDemented',
           'MildDemented',
           'ModerateDemented']

IMG_SIZE = 176
IMAGE_SIZE = [176, 176]
DIM = (IMG_SIZE, IMG_SIZE)

"""
Here we perform our data augmentation by using some types of change to our images, such
as Zooming, Bright, Flipping
"""
ZOOM = [.99, 1.01]
BRIGHT_RANGE = [0.8, 1.2]
HORZ_FLIP = True
FILL_MODE = "constant"
DATA_FORMAT = "channels_last"

work_dr = IDG(rescale=1. / 255, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM, data_format=DATA_FORMAT,
              fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP)

"""
Create our training data using tensorflow and keras modules
"""
train_data_gen = work_dr.flow_from_directory(directory=WORK_DIR, target_size=DIM, batch_size=6500, shuffle=False)

"""
Configure image, label iterator
"""
train_data, train_labels = train_data_gen.next()


"""
Initialize Smote for data augmentation and applying augmentation on our data
"""
sm = SMOTE(random_state=42)

train_data, train_labels = sm.fit_resample(train_data.reshape(-1, IMG_SIZE * IMG_SIZE * 3), train_labels)

train_data = train_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

"""
Splitting the data into train, test, and validation sets
"""
train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2,
                                                                    random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2,
                                                                  random_state=42)

"""
From here we finish the pre-processing and now we focus on the CNN model
"""
inception_model = InceptionV3(input_shape=(176, 176, 3), include_top=False, weights="imagenet")

for layer in inception_model.layers:
    layer.trainable = False

"""
Building the CNN model layer by layer 
"""
custom_inception_model = Sequential([
    inception_model,
    Dropout(0.5),
    GlobalAveragePooling2D(),
    Flatten(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(4, activation='softmax')
], name="inception_cnn_model")


"""
Function to stop training if it reaches over 0.99 accuracy
"""
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.99:
            print("\nReached accuracy threshold! Terminating training.")
            self.model.stop_training = True


my_callback = MyCallback()

"""
ReduceLROnPlateau to stabilize the training process of the model
"""
rop_callback = ReduceLROnPlateau(monitor="val_loss", patience=3)
"""
Here we choose the metrics to evaluate our model using AUC-ROC and Accuracy
"""
METRICS = [tf.keras.metrics.CategoricalAccuracy(name='acc'),
           tf.keras.metrics.AUC(name='auc')]

CALLBACKS = [my_callback, rop_callback]

"""
Here we compile our model and optimize it using rmsprop also choosing our loss function
as Categorical Cross-Entropy
"""
custom_inception_model.compile(optimizer='rmsprop',
                               loss=tf.losses.CategoricalCrossentropy(),
                               metrics=METRICS)
"""
Checking the everything is correct
"""
custom_inception_model.summary()

"""
Fit the training data to the model and validate it using the validation data
here we set number of Epochs
"""

EPOCHS = 25

history = custom_inception_model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                                     callbacks=CALLBACKS, epochs=EPOCHS)

"""
Plot our results
"""
fig, ax = plt.subplots(1, 3, figsize=(30, 5))
ax = ax.ravel()

for i, metric in enumerate(["acc", "auc", "loss"]):
    ax[i].plot(history.history[metric])
    ax[i].plot(history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("Epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
plt.show()


test_scores = custom_inception_model.evaluate(test_data, test_labels)


print("Testing Accuracy: %.2f%%" % (test_scores[1] * 100))
# Predicting the test data

pred_labels = custom_inception_model.predict(test_data)


# Print the classification report of the tested data

# Since the labels are softmax arrays, we need to roundoff to have it in the form of 0s and 1s,
# similar to the test_labels
def roundoff(arr):
    """To round off according to the argmax of each predicted label array. """
    arr[np.argwhere(arr != arr.max())] = 0
    arr[np.argwhere(arr == arr.max())] = 1
    return arr


for labels in pred_labels:
    labels = roundoff(labels)

print(classification_report(test_labels, pred_labels, target_names=CLASS))

# Plot the confusion matrix to understand the classification in detail

pred_ls = np.argmax(pred_labels, axis=1)
test_ls = np.argmax(test_labels, axis=1)

conf_arr = confusion_matrix(test_ls, pred_ls)

plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

ax = sns.heatmap(conf_arr, cmap='Blues', annot=True, fmt='d', xticklabels=CLASS,
                 yticklabels=CLASS)

plt.title('Alzheimer\'s Disease Diagnosis')
plt.xlabel('Prediction')
plt.ylabel('Truth')
plt.show()
