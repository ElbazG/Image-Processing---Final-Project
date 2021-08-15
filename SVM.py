import pandas as pd
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

"""

Hyper-Parameters:
1. IMG_SIZE - Size of images to the SVM
2. DIRECTORY - The directory that stores all our Dataset images
3. CLASS - Our classifications options for each image

"""

IMG_SIZE = 200
DIRECTORY = "C:/Users/Gil/PycharmProjects/img_process/Union"
CLASS = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

plt.figure(figsize=(20, 20))

"""
Create a list to store all our training data
"""

training_data = []

"""
Function to get the data from the given directory
and iterate it.
We index our categories to 0,1,2,3
We append to separate lists an image and its class
Finally we add a tuple containing (Image,Class) to our training data list 
"""


def create_training_data():
    for classification in CLASS:
        path = os.path.join(DIRECTORY, classification)
        class_num = CLASS.index(classification)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()

count = len(training_data)
"""
In order to feed the SVM we need to put each one in separate vector (array)
We used numpy to convert it into an array.
X - Images
Y - Label (Class)
"""
X = []
y = []

for image, label in training_data:
    X.append(image)
    y.append(label)

X = np.array(X).reshape(count, -1)

"""
Flatten the array
"""
X = X / 255.0

y = np.array(y)
"""
We create new variables to hold train and test images and their corresponding labels.
We use train_test_split function to split our data by default - 75% Train, 25% Test.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y)

"""
Build the model using linear kernel, then train it using fit()
then after training it predict on our test data and put it on y2.
"""
svc = SVC(kernel='linear', gamma='auto')
svc.fit(X_train, y_train)

y2 = svc.predict(X_test)

print("Accuracy on unknown data is", accuracy_score(y_test, y2))

print("Accuracy on unknown data is", classification_report(y_test, y2))

result = pd.DataFrame({'original': y_test, 'predicted': y2})
