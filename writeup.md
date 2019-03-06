# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model

* run1.mp4 showing my vehicle driving autonomously for  one lap

* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and 24 filters (code line 145)
then a convolution neural network with 5x5 filter sizes and 36 filters
then a convolution neural network with 5x5 filter sizes and 48 filters
then  a convolution neural network with 3x3 filter sizes and 64 filters
then  a convolution neural network with 3x3 filter sizes and 64 filters
then fully connected layers of 100 then 50 then 10 and output layer of 1 .

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer (code line 143). 

#### 2. Attempts to reduce overfitting in the model

the model is trained on big data set contains images of the track  for multiple lap to reduce overfitting also the data was augmented by fliping and using multiple camera cretria
 The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 156.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving,also I used data from both sides 
trying to get to the center, I made this by driving the car towards the sides without recording then starts recording from the sides trying to recovery and get to the center

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make a model perdicts the right angle depending on the positon of the car with respect to the lanes.

My first step was to use a convolution neural network model similar to the lenet architecture I thought this model might be appropriate because it works good for images classifing 
but it doesn't work well so I decided to use an architecture simlar to that was used in the project lesson.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. and the was low on both trainig and validation set and it decreases 
every epoch for the first 3 epoch then in the next epochs the training set error decreses while the validation set increase that indicates that overfitting has occured.


To combat the overfitting, I modified the model so that it trains only for 3 epochs


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I increased the recovery data to make the car returns to road when it leaves the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture was the same archtecture that I explained perviously.

