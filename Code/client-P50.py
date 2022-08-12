from collections import OrderedDict
from IPython.display import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas
import cv2 as cv
from numpy import expand_dims
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from matplotlib import pyplot
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation, GlobalAveragePooling2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

import flwr as fl

toTERFolder = Path().absolute()
testData30 = toTERFolder + '/TestData/P30'
testData50 = toTERFolder + '/TestData/P50'
data30 = toTERFolder + '/DataClient2-P50/P30'
data50 = toTERFolder + '/DataClient2-P50/P50'
modelPath = toTERFolder + '/DataClient1-P50/model'

classes = ['Speed Limit 30', 'Speed Limit 50']
nbClasses = 2
classLabel = 0
nbAugmented = 10
xTrain = np.empty(shape=(0, 224, 224, 3))
yTrain = []
xTest = np.empty(shape=(0, 224, 224, 3))
yTest = []

for cl in classes:
    augmentedIndex=0
    if(cl == 'Speed Limit 30'):
        listImages = glob.glob(data30+'/*')
    else:
        listImages = glob.glob(data50+'/*')
    yTrain += [classLabel]*len(listImages)*(nbAugmented + 1)
    for pathImg in listImages:
        img = image.load_img(pathImg, target_size=(224, 224))
        im = image.img_to_array(img)
        normalizedImg = np.zeros((224, 224))
        im = cv.normalize(im,  normalizedImg, 0, 255, cv.NORM_MINMAX)
        im = np.expand_dims(im, axis=0)
        im = preprocess_input(im)
        xTrain = np.vstack([xTrain, im]) #Adding normal image
        datagen = ImageDataGenerator(rotation_range=90, zoom_range=[0.5,1.0])
        it = datagen.flow(im, batch_size=1)
        for new in range(nbAugmented):
            batch = it.next()
            im = np.expand_dims(batch[0], axis=0)
            if(cl == 'Speed Limit 30'):
                cv.imwrite(data30 + '/augmented' + str(augmentedIndex) + '.jpg', batch[0])
            else:
                cv.imwrite(data50 + '/augmented' + str(augmentedIndex) + '.jpg', batch[0])
            xTrain = np.vstack([xTrain, im]) #Adding transformed image
            augmentedIndex = augmentedIndex + 1
    classLabel += 1

classLabel = 0
for cl in classes:
    if(cl == 'Speed Limit 30'):
        listImages = glob.glob(testData30+'/*')
    else:
        listImages = glob.glob(testData50+'/*')
    yTest += [classLabel]*len(listImages)
    for pathImg in listImages:
        img = image.load_img(pathImg, target_size=(224, 224))
        im = image.img_to_array(img)
        normalizedImg = np.zeros((224, 224))
        im = cv.normalize(im,  normalizedImg, 0, 255, cv.NORM_MINMAX)
        im = np.expand_dims(im, axis=0)
        im = preprocess_input(im)
        xTest = np.vstack([xTest, im]) #Adding normal image
    classLabel += 1

yTrain = keras.utils.to_categorical(yTrain, nbClasses)
xTrain, yTrain = shuffle(xTrain, yTrain)

yTest = keras.utils.to_categorical(yTest, nbClasses)
xTest, yTest = shuffle(xTest, yTest)

# Load model and data
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1), padding='valid'))
model.add(Flatten())
model.add(Dense(nbClasses, activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(xTrain, yTrain, epochs=15, batch_size=32, steps_per_epoch=3) #first train

model.save(modelPath)

print("Should have bad results on P30")
yPredictedTest = model.predict(xTest)
print(confusion_matrix(yTest.argmax(axis=1), yPredictedTest.argmax(axis=1)))
print(f1_score(yTest.argmax(axis=1), yPredictedTest.argmax(axis=1)))
print(accuracy_score(yTest.argmax(axis=1), yPredictedTest.argmax(axis=1)))

class PanelClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(xTrain, yTrain, epochs=15, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(xTrain), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(xTest, yTest)
        return loss, len(xTest), {"accuracy": accuracy}

print("\nFederation")
# Load Flower
fl.client.start_numpy_client("[::]:8080", client=PanelClient())

print("\nShould have good results on P30")
yPredictedTest = model.predict(xTest)
print(confusion_matrix(yTest.argmax(axis=1), yPredictedTest.argmax(axis=1)))
print(f1_score(yTest.argmax(axis=1), yPredictedTest.argmax(axis=1)))
print(accuracy_score(yTest.argmax(axis=1), yPredictedTest.argmax(axis=1)))