import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

IMG_SIZE = 224

##TRAIN_DIR = 'D:\\08\\Final Year Project Implementation\\Raw Dataset\\Trials\\'

def label_img(img):
    word_label = img.split('_')[-2]
    if word_label == "500real" or word_label == '2000real':
        return [1, 0]
    elif word_label == '500fake' or word_label == '2000fake':
        return [0, 1]


def create_train_data():
    training_data = []
    for TRAIN_DIR in ['D:\\08\\Final Year Project Implementation\\Raw Dataset\\500_Real\\', 'D:\\08\\Final Year Project Implementation\\Raw Dataset\\500_Fake\\']:
        for img in tqdm(os.listdir(TRAIN_DIR)):
            label = label_img(img)
            path = os.path.join(TRAIN_DIR, img)
            
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            lowThresh = 0.5*high_thresh
            blur = cv2.medianBlur(gray, 17)
            
            edge_img = cv2.Canny(blur,lowThresh*1.5,high_thresh*1.5)
            pts = np.argwhere(edge_img>0)
            try:
                y1,x1 = pts.min(axis=0)
                y2,x2 = pts.max(axis=0)
            except ValueError:
                print (img)
                pass
            
            cropped = gray[y1:y2, x1:x2]
            blur_cropped = cv2.medianBlur(cropped, 7)
            blur_cropped = cv2.resize(blur_cropped , (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(blur_cropped), np.array(label)])
            
            #Applying Salt and Pepper filter
            s_vs_p = 0.5
            amount = 0.02
            for j in range(0,5):
                sp_img = np.copy(blur_cropped)
                # Salt mode
                num_salt = np.ceil(amount * blur_cropped.size * s_vs_p)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in blur_cropped.shape]
                sp_img[coords] = 255
                # Pepper mode
                num_pepper = np.ceil(amount* blur_cropped.size * (1. - s_vs_p))
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in blur_cropped.shape]
                sp_img[coords] = 0
                training_data.append([np.array(sp_img), np.array(label)])
            
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

train_data = create_train_data()
print(len(train_data))
