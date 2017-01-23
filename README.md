# Behavioral cloning project
============================

## Overview of project
### Model architecture
I initially  used the VGG architecture to train my model, but it took very long to train and it didn't quite hit the mark. After a bit of reasearch I found that the Nvidia model is a good alternative, so I used that instead. The Nvidia model expects an image with 66 rows and 200 columns, but Keras is nice enough to reshape the images to fit.

Below is a detailed table of the model I used. It had 22 layers in total, and the size of each layer is detailed below. 

|Layer (type)	           | Output Shape       |
|:---------------------:|:-------------------:|
|lambda_1 (Lambda)      |	(None, 93, 272, 3)  |
|conv1 (Convolution2D)  | (None, 47, 136, 24) |
|dropout_1 (Dropout)    |	(None, 47, 136, 24) |
|elu_1 (ELU)            | (None, 47, 136, 24) |
|conv2 (Convolution2D)	| (None, 24, 68, 36)  |
|elu_2 (ELU)	          | (None, 24, 68, 36)  |
|conv3 (Convolution2D)	| (None, 10, 32, 48)  |
|elu_3 (ELU)	          | (None, 10, 32, 48)  |
|conv4 (Convolution2D)	| (None, 8, 30, 64)   |
|elu_4 (ELU)	          | (None, 8, 30, 64)   |
|conv5 (Convolution2D)	| (None, 6, 28, 64)   |
|dropout_2 (Dropout)	  | (None, 6, 28, 64)   |
|elu_5 (ELU)	          | (None, 6, 28, 64)   |
|flatten_1 (Flatten)	  | (None, 10752)       |      
|hidden1 (Dense)	      | (None, 100)         |   
|elu_6 (ELU)	          | (None, 100)         |
|hidden2 (Dense)	      | (None, 50)          |
|elu_7 (ELU)	          | (None, 50)          |
|hidden3 (Dense)       	| (None, 10)          |
|elu_8 (ELU)	          | (None, 10)          |
|output (Dense)        	| (None, 1)           |

### Data gathering
In terms of gathering data, I have two datasets. The first is just a collection of data from 3 laps with normal driving. The second data set is for recovery, where I intentionally drove to the sides of the road to turn back. The idea is to teach the model how to react when it gets too close to the sides. I used an Xbox controller as input, which made life much easier (finally justified the purchase of it :))

### Data handling
Most of my time was spent on trying to get the model to use the right data. This included under-sampling zero-driving angles (I eventually just dropped all data with exactly zero driving angles), balancing left and right turns by mirroring images, and using all three cameras (with an appropriate  offset). 

### Image processing
For image processing, I eventually opted only to use brightness - valuable information I got from Slack. Initially I normalized the images and turned it into grayscale, but it added little value. I think that by grayscaling an image you lose too much information - the red/yellow stripes, the dirt track parts and so on. I still normalize my data, but as an additional layer in my model - not as preprocessing.

Below are image examples, after being cropped, brightness changed and resized 
![Centre](/images/centre.png)

Left camera images had an adjument of +0.25 made to the driving angle
![Left](/images/left.png)

While right camerae images an adjustment of -0.25 
![Right](/images/right.png)



## Output
[![IMAGE ALT TEXT](http://img.youtube.com/vi/962B7emgbGI/0.jpg)](https://www.youtube.com/watch?v=962B7emgbGI "Driving Ms Daisy")

## Difficulties
I would say 99.9993% of the project is about feeding the right data. Not so much what your simulation looks like (it's still important), but what you do with it. Almost all of the problems I solved, was only after _looking_ at my processed images, checking what data was used for the training, if it uses a balanced set between the different cameras, does it over/ or under-sample and so on.

If any other students bothered to read this far, use Slack. No really....just read through it if you feel stuck. There are amazing people on the channel who will be happy to help when they can. 

If I could go back in time and give myself some tips:
* There's a difference between random's shuffle and sklearn's shuffle
* np arrays are awesome, use it
* TF works in windows now, don't bother with Docker
* Use slack
* Flip **after** adjusting angles for cameras

