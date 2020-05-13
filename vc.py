# in this file the code for video classification has been given. in this we have taken short span of videos to identify the type of activity 
# that is being done. the video has been divided into various frames and we have done image classification on those frames to identify
# the activity

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2
import glob
import math
import pickle

# Getting Labelled 
names = os.listdir("C:/Users/GOEL/CHAITANYA/WINTER TRAINING JAN 2020/UCF11/dataset")

# Extracting images from each video of every category
for name in tqdm(names):
    count = 0
    a = glob.glob(''+name+'/*.mpg')
    for i in range(len(a)):
        cap = cv2.VideoCapture(a[i])
        frameRate = cap.get(5)
        while(cap.isOpened()):
            frameId = cap.get(1)
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                cv2.imwrite('dataset/'+name+'/'+'{}.jpg'.format(count), frame)
                count += 1
        cap.release()

import glob

images = [cv2.imread(file) for file in glob.glob("C:/Users/GOEL/CHAITANYA/WINTER TRAINING JAN 2020/UCF11/dataset/swing/*.jpg")]

images_1 = [cv2.imread(file) for file in glob.glob("C:/Users/GOEL/CHAITANYA/WINTER TRAINING JAN 2020/UCF11/dataset/walking/*.jpg")]


mera_dat = []

for i in range(250):
    desired_size = 368
    
    im = images[i]
    old_size = im.shape[:2] # old_size is in (height, width) format
    
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # new_size should be in (width, height) format
    
    im = cv2.resize(im, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
#    cv2.imshow("image", new_im)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
#    cv2.imwrite('C:/Users/Rahul/Desktop/a.jpg'.format(i), new_im) 
    mera_dat.append(new_im)


mera_dat_1 = []

for i in range(250):
    desired_size = 368
    
    im = images_1[i]
    old_size = im.shape[:2] # old_size is in (height, width) format
    
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # new_size should be in (width, height) format
    
    im = cv2.resize(im, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
#    cv2.imshow("image", new_im)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
#    cv2.imwrite('C:/Users/Rahul/Desktop/a.jpg'.format(i), new_im) 
    mera_dat_1.append(new_im)


arr = np.array(mera_dat)
ar1 = np.array(mera_dat_1)

arr = arr.reshape((250, 406272))
ar1 = ar1.reshape((250, 406272))

arr = arr / 255
ar1 = ar1 / 255

dataset = pd.DataFrame(arr)
dataset['label'] = np.ones(250)

dataset.iloc[:, -1]

dataset_1 = pd.DataFrame(ar1)
dataset_1['label'] = np.zeros(250)

dataset_1.iloc[:, -1]

dataset_master = pd.concat([dataset, dataset_1])


dataset_master.iloc[:, 406272]

X = dataset_master.iloc[:, 0:406272].values
y = dataset_master.iloc[:, -1].values

import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential()
model.add(keras.layers.Dense(256, activation = 'relu'))
model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dense(3, activation = 'softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(X, y, epochs = 5)

pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()














































