#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia-architecture.png "Nvidia architecture"
[image2]: ./examples/model.png "Final model"
[image3]: ./examples/center_lane.jpg "Center lane"
[image4]: ./examples/recovery_lane_1.jpg "Recovery lane 1"
[image5]: ./examples/recovery_lane_2.jpg "Recovery lane 2"
[image6]: ./examples/recovery_lane_3.jpg "Recovery lane 3"
[image7]: ./examples/recovery_lane_4.jpg "Recovery lane 4"
[image8]: ./examples/recovery_lane_5.jpg "Recovery lane 5"
[image9]: ./examples/before_flip.png "Before flipping"
[image10]: ./examples/after_flip.png "After flipping"
[image11]: ./examples/before_trans.png "Before translation"
[image12]: ./examples/after_trans.png "After translation"



###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used the Nvidia architecture (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), which was considered in the lessons.

![Nvidia architecture][image1]

The model consists of three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers (model.py lines 116-133). 
As an activation function I used "ELU". In addition, I added cropping and 3 lambda layers (grayscale, resize(32,32), normalize(-1,1)). This will ensure that the model will preprocess input images when making predictions (model.py lines 112-115). Cropping will only leave useful information. Converting to grayscale and resizing(32, 32) of input data to train faster.

####2. Attempts to reduce overfitting in the model

The model contains l2 regularization for all convolution and fully-connected layers in order to reduce overfitting.
The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer with the learning rate 0.0001 (model.py line 135), epochs equal to 10 and batch size equal to 32.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (3 laps), recovering from the left and right sides of the road (2 laps) for the track one. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I used the Nvidia architecture, which was considered in the lessons. To speed up the training and reduce the amount of gpu memory needed, I first cropped of input data, then converted to grayscale and resized to 32x32 pixels. As an activation function I used "ELU". To combat the overfitting, I used l2 regularization for all convolution and fully-connected layers, and splitted training and validation data by "train_test_split" function. Validation data is 10% of all samples. 

At the end of the process, the vehicle is able to drive autonomously around the track one without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 116-133) consisted of three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers.

Here is a visualization of the architecture (model visualization was obrained by keras function "plot_model")

![Final model][image2]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![Center lane][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center (2 laps) so that the vehicle would learn to recover back to center if it veers off to the side. These images show what a recovery looks like:

![Recovery lane 1][image4]
![Recovery lane 2][image5]
![Recovery lane 3][image6]
![Recovery lane 4][image7]
![Recovery lane 5][image8]

I used images from all three cameras, if steering angle was in the interval [-0.75; 0.75]. Correction factor of an angle for right and left images is equal to 0.25.
To augment the data set, I also flipped (model.py lines 81-83) and translated (model.py lines 34-43 and 76-78) data thinking that this would to reduce overfitting and allow to generalize the model.

Here is an image that has then been flipped:

![Before flipping][image9]
![After flipping][image10]

Here is an image that has then been translated:

![Before translation][image11]
![After translation][image12]

After the collection process, I had 8724 lines in "driving_log.csv" file. These lines were randomly shuffled and put 10% of the data into a validation set by function "train_test_split".
Also I used generator (model.py lines 46-87) which pulls pieces of the data and augments data by translation and flipping functions.
