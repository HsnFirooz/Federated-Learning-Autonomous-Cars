from IPython.display import Image
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

data30 = '/Users/hbp/Documents/GitHub.nosync/TER2021-074/Documentation/30-PANEL-PR/'
data50 = '/Users/hbp/Documents/GitHub.nosync/TER2021-074/Documentation/50-PANEL-PR/'
classes = ['Speed Limit 30', 'Speed Limit 50']
nbClasses = 2
classLabel = 0
nbAugmented = 10
xTrain = np.empty(shape=(0, 224, 224, 3))
yTrain = []

for cl in classes:
    print(cl)
    
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
            xTrain = np.vstack([xTrain, im]) #Adding transformed image
    
    classLabel += 1

yTrain = keras.utils.to_categorical(yTrain, nbClasses)
xTrain, yTrain = shuffle(xTrain, yTrain)
amountOfTrain = round(len(xTrain) * 0.7)

xTest = xTrain[amountOfTrain:]
yTest = yTrain[amountOfTrain:]
xTrain = xTrain[:amountOfTrain]
yTrain = yTrain[:amountOfTrain]

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1), padding='valid'))
model.add(Flatten())
model.add(Dense(nbClasses, activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(xTrain, yTrain, epochs=30, batch_size=128)

yPredictedTest = model.predict(xTest)

print(yPredictedTest)
print(yTest)
print(confusion_matrix(yTest.argmax(axis=1), yPredictedTest.argmax(axis=1)))
print(f1_score(yTest.argmax(axis=1), yPredictedTest.argmax(axis=1)))
print(accuracy_score(yTest.argmax(axis=1), yPredictedTest.argmax(axis=1)))







