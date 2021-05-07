import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

IMG_SIZE = 264
noOfCols = 264
noOfRows = 264
TRAIN_DIR = '' #add path


def label_img(img):
    word_label = img.split('_')[-2]
    if word_label == "500real" or word_label == '2000real':
        return [1, 0]
    elif word_label == '500fake' or word_label == '2000fake':
        return [0, 1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.medianBlur(img, 5)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        for angle in range(15, -16, -1):
            rotationMatrix = cv2.getRotationMatrix2D((noOfCols/2, noOfRows/2), angle, 1)
            imgRotated = cv2.warpAffine(img, rotationMatrix, (noOfCols, noOfRows))
            training_data.append([np.array(imgRotated), np.array(label)])
    np.save('train_data.npy', training_data)
    return training_data

train_data = create_train_data()
