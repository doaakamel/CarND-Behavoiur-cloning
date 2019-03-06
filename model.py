import os
import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten , Dense, Lambda
from keras.layers.convolutional import Convolution2D,Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dropout
from keras.models import load_model

###############################loading data ##########################################################################################
lines =[]
images =[]
measurments =[]
flag=0
with open ("./recovery9/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines:
    if flag==0:
        flag=1
        continue 
    source_path=line[0]
    filename= source_path.split('/')[-1]
    current_path="./recovery9/IMG/" +filename 
    image=cv2.imread(current_path)
    images.append(image)
    measurment = float(line[3])
    measurments.append(measurment)
    images.append(cv2.flip(image,1))
    measurments.append(measurment*-1.0)
lines=[]
with open ("./recovery7/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines:
    if flag==0:
        flag=1
        continue 
    source_path=line[0]
    filename= source_path.split('/')[-1]
    current_path="./recovery7/IMG/" +filename 
    image=cv2.imread(current_path)
    images.append(image)
    measurment = float(line[3])
    measurments.append(measurment)
    images.append(cv2.flip(image,1))
    measurments.append(measurment*-1.0)

lines=[]   
with open ("./recovery6/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
for line in lines:
    if flag==0:
        flag=1
        continue 
    source_path=line[0]
    filename= source_path.split('/')[-1]
    current_path="./recovery6/IMG/" +filename 
    image=cv2.imread(current_path)
    images.append(image)
    measurment = float(line[3])
    measurments.append(measurment)
    images.append(cv2.flip(image,1))
    measurments.append(measurment*-1.0)
lines=[]   
with open ("./recovery8/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
for line in lines:
    if flag==0:
        flag=1
        continue 
    source_path=line[0]
    filename= source_path.split('/')[-1]
    current_path="./recovery8/IMG/" +filename 
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
model.add(Dropout(0.7))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=8)

model.save("model.h5")
print("done")
exit()
