#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode, the speed is updated to 18
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results
* `generate_data.py` used for generating additional training data through
 autonomous driving
* a YouTube video of the autonomous driving using the model on the 
second track 

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model (defined in the function `train_model`, lines 98-139 in `model.py`) is a convolutional neural network based on the 
[Nvidia's convolutional neural network for self-driving](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
It consists of 5 convolutional layers followed by 4 fully-connected layers. Relu activation is applied between each layer, and we also apply
dropout between final convolutional layer and the first fully-connected layer to control overfitting.

Please refer to `model.py` for the exact dimension of each layer.

####2. Attempts to reduce overfitting in the model

As mentioned in (1.), the model contains a dropout layer between the final convolutional layer
 and the first fully-connnected layer in order to reduce overfitting (`model.py` line 132). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (`model.py` lines 149-152). 
We also used checkpoints and early-stopping (`model.py` lines 144-147) to retain the highest performing models (based on validation loss) and stop
the training early if the validation loss does not improve in three epochs. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` line 141).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I drove the car both clockwise and 
counter-clockwise on both tracks while trying to keep the car near the center of the lane.

For additional details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to gather a large set of "high quality" (i.e. well-driving) training data and 
modify Nvidia's convolution neural network architecture to our data dimension. I chose use Nvidia's architecture partly because it
was recommended in the course video, partly because it's fairly simple to implement and I was curious to see how well it does.

Given the dimension of the training images is 90x230x3 and not 66x200x3 as in Nvidia's architecture, I modified the filters and strides to 
account for the increased dimensions. I refer the readers to `model.py` for the detailed architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set at a 80/20 ratio
by setting the `validation_split` parameter in `model.fit` (line 148) to 0.2. 

In order to combat overfitting, I added a dropout layer with 0.5 probability of dropping a weight after the flattening (line 132, between the final
convolutional layer and the first fully-connected layer). Moreover, I enabled checkpoint and early-stopping callbacks to save the best performing
model so far (based on validation loss) and terminate the training early if the validation loss does not improve in 3 epochs (lines 144-147).

The final step was to run the simulator to see how well the car was driving around track one. 
There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (`model.py` lines 98-139) consisted of a convolution neural network with the following layers and layer sizes:

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...

For track 2, I also drove both clockwise and counter-clockwise around the track, while trying
to stay between the lane lines on the right side. Then I repeated the clockwise and counter-clockwise
drives but tried to stay between the lane lines on the left side. This gave me four sets of training
data for track 2.

Finally, after training an initial model using these data, I modified `drive.py` to generate
more training data through autonomous driving. The updated code is in `generate_data.py`. I used
a speed limit of 20 to generate the new data.

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
