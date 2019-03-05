import os
import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten , Dense, Lambda
from keras.layers.convolutional import Convolution2D,Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model

###########################################################loading the data##############################################################################
lines =[]
with open ("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
images =[]
measurments =[]
flag=0
for line in lines:
    if flag==0:
        flag=1
        continue 
    source_path=line[0]
    filename= source_path.split('/')[-1]
    current_path="./data/IMG/" +filename 
    image=cv2.imread(current_path)
    images.append(image)
    measurment = float(line[3])
    measurments.append(measurment)
    images.append(cv2.flip(image,1))
    measurments.append(measurment*-1.0)


with open ("./recovery/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines=[]
for line in lines:
    if flag==0:
        flag=1
        continue 
    source_path=line[0]
    filename= source_path.split('/')[-1]
    current_path="./recovery/IMG/" +filename 
    image=cv2.imread(current_path)
    images.append(image)
    measurment = float(line[3])
    measurments.append(measurment)
    images.append(cv2.flip(image,1))
    measurments.append(measurment*-1.0)
    
    
 ######################################################################## loading recovery data for the cases the car goes left ##################################################   
    
with open ("./recovery1/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines=[]
for line in lines:
    if flag==0:
        flag=1
        continue 
    source_path=line[0]
    filename= source_path.split('/')[-1]
    current_path="./recovery1/IMG/" +filename 
    image=cv2.imread(current_path)
    images.append(image)
    measurment = float(line[3])
    measurments.append(measurment)
    images.append(cv2.flip(image,1))
    measurments.append(measurment*-1.0)
    
#############################################################loading recovery data for  the cases the car goes right ##############################################################
with open ("./recovery2/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines=[]
for line in lines:
    if flag==0:
        flag=1
        continue 
    source_path=line[0]
    filename= source_path.split('/')[-1]
    current_path="./recovery2/IMG/" +filename 
    image=cv2.imread(current_path)
    images.append(image)
    measurment = float(line[3])
    measurments.append(measurment)
    images.append(cv2.flip(image,1))
    measurments.append(measurment*-1.0)
###############################################################3using multiple cameras ##########################################################################################
with open ("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines=[]
for line in lines:
    if flag==0:
        flag=1
        continue 
    source_path=line[1]
    filename= source_path.split('/')[-1]
    current_path="./data/IMG/" +filename 
    image=cv2.imread(current_path)
    images.append(image)
    measurment = float(line[3])
    measurments.append(measurment)
    images.append(cv2.flip(image,1))
    measurments.append(measurment*-1.0)

with open ("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines=[]
for line in lines:
    if flag==0:
        flag=1
        continue 
    source_path=line[2]
    filename= source_path.split('/')[-1]
    current_path="./data/IMG/" +filename 
    image=cv2.imread(current_path)
    images.append(image)
    measurment = float(line[3])
    measurments.append(measurment)
    images.append(cv2.flip(image,1))
    measurments.append(measurment*-1.0)
              
#######################################add the images to traiin array and shuffle the data #########################################################################################
X_train=np.array(images)
Y_train=np.array(measurments)
sklearn.utils.shuffle(X_train, Y_train)

######################################################## Model Architecture #########################################################################################################
model = Sequential() 

model.add(Lambda (lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,subsample=(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save("model.h5")
exit()