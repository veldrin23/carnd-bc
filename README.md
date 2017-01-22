# Behavioral cloning project
============================

## Overview of project
### Model architecture
I initially  used the VGG architecture to train my model, but it took very long to train and it didn't quite hit the mark. After a bit of reasearch I found that the Nvidia model is a good alternative, so I used that instead. The Nvidia model expects an image with 66 rows and 200 columns, but Keras is nice enough to reshape the images to fit.

### Data gathering
In terms of gathering data, I have two datasets. The first is just a collection of data from 3 laps with normal driving. The second data set is for recovery, where I intentionally drove to the sides of the road to turn back. The idea is to teach the model how to react when it gets too close to the sides. I used an Xbox controller as input, which made life much easier (finally justified the purchase of it :))

### Data handling
Most of my time was spent on trying to get the model to use the right data. This included under-sampling zero-driving angles (I eventually just dropped all data with exactly zero driving angles), balancing left and right turns by mirroring images, and using all three cameras (with an appropriate  offset). 

### Image processing
For image processing, I eventually opted only to use brightness - valuable information I got from Slack. Initially I normalized the images and turned it into grayscale, but it added little value. I think that by grayscaling an image you lose too much information - the red/yellow stripes, the dirt track parts and so on. I still normalize my data, but as an additional layer in my model - not as preprocessing.

## Difficulties
I would say 99.9993% of the project is about feeding the right data. Not so much what your simulation looks like (it's still important), but what you do with it. Almost all of the problems I solved, was only after _looking_ at my processed images, checking what data was used for the training, if it uses a balanced set between the different cameras, does it over/ or under-sample and so on.

If any other students bothered to read this far, use Slack. No really....just read through it if you feel stuck. There are amazing people on the channel who will be happy to help when they can. 

If I could go back in time and give myself some tips:
* There's a difference between random's shuffle and sklearn's shuffle
* np arrays are awesome, use it
* TF works in windows now, don't bother with Docker
* Use slack
* Flip after adjusting angles for cameras
