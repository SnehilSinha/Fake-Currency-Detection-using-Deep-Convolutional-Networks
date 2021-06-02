import numpy as np
import matplotlib as pyplot
import os
import cv2
import random

def create_training_data():
    for category in CATEGORIES:
        path= os.path.join(DATADIR,category)
        c_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            image_arr=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_arr=cv2.resize(image_arr,(100,100))
            training_data.append([new_arr,c_num])

DATADIR = "Datasets/Notes"
CATEGORIES =["Real","Fake"]
IMG_SIZE = 200
training_data=[]

create_training_data()

random.shuffle(training_data)

X=[]
y=[]

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,100,100,1)
y = np.array(y)
X=X/255
#print(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))                        #SINGLE OUTPUT node
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy" , optimizer="rmsprop" , metrics=['accuracy'])   #optimizer=adam
model.fit(X_train,y_train, batch_size=32 ,verbose=1,epochs=50,validation_data=(X_test,y_test),shuffle=False)

