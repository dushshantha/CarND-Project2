# Traffic Sign Recognition

## Writeup Template


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images/HistBefore.png "Histogram"
[image2]: ./Images/Color.png "Color"
[image3]: ./Images/Gray.png "Gray"
[image4]: ./Images/Scaled.png "Scaled"
[image5]: ./Images/Warped.png "Warped"
[image6]: ./Images/HistAfter.png "Histogram"
[image7]: ./WebImages/1.png "Traffic Sign 1"
[image8]: ./WebImages/2.png "Traffic Sign 2"
[image9]: ./WebImages/3.png "Traffic Sign 3"
[image10]: ./WebImages/4.png "Traffic Sign 4"
[image11]: ./WebImages/5.png "Traffic Sign 5"
[image12]: ./WebImages/5.png "Traffic Sign 6"
[image13]: ./Images/predictions.png "predictions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Data Set Summary & Exploration

Below is a summary statistics of the data set. numpy provides a set of operations to easily catpure this data. numpy.ndarray.shape is used in gettig the dimentions of the dataset listed below and numpy.unique privides an easy way to count the unique number of items in a list.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization of the dataset.

Here is an the Histogram of Classes and number of samples from each class in the traning data set. Data set is pretty imbalance. During pre-processing, I am trying to get this close to a balance for a better performance of the model. 

![alt text][image1]

#### Design and Test a Model Architecture

##### 1. Pre-Processing of Data

As Preprocessing of the images, I used few steps that I decribe below. 

###### 1. Grayscale convertion.

Converting the images to Grayscale reduces the complexity of the data tremendouly. This improves the training of the model easier and more efficient. 

Below is an example image before the grayscale.

![alt text][image2]

Here is an example of a traffic sign image after grayscaling.

![alt text][image3]

###### 2. Normalization

The 2nd pre processing step is to normalize the data. The mean of the Original training data set before the Normalization was 81.9172385241. I applied (X- 255) / 255 to bring this number down to somewhere in between 1 and -1. After normalization, the mean of the Traning data set came down to 0.297134071736

###### 3. Augmentation

As I mentioned earlier, the training data is not balanced amung the classes. There are come classes that has a huge amount of samples compared some other classes. This leads to inaccurate models. In order to correct this, I applied few image augmentation techniques to generate some additional Training data for the classes that does not have at least the average amount of samples. I used two techniques here. 
1. Random Scaing
2. Random Warp. 

I used these 2 combined on random samples of classes and added to the traning set. This increased the number of Training data set from 34799 to 46480.

Here's an example of Randon Scale.

![alt text][image4]

Here's an example of Random Warp

![alt text][image5]

This process produced the following histogram. As you can see, this new traning Set is significanly more balanced than the original set. 

![alt text][image6]


##### 2. The Model
Please refer [Sermanet & LeCunn](http://yann.lecun.org/exdb/publis/psgz/sermanet-ijcnn-11.ps.gz) for the architecture. 
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 10x10x16								|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten					|Output = 400												|
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 1x1x400							|
| Flatten					|Output = 800												|
|	Dropout					|	Keep_prob = 0.5											|
|	Fully connected					|	Output = 43										|
 


#### 3. Training the Model

The model was training using AdamOptimer with a Batch size of 100 and 60 epoches. I experimented with 0.001, 0.0007, 0.0008 and finally 0.0009 as the learning rate. 

#### 4. The Solution
I started with the LeNet model architecture We built as part of the Lenet lab in Udacity Self Driving Car Nano Degree program. I then changed the architecture to match  Sermanet & LeCunn's layered architecture. The link to the paper can be found in the previous section "The Model". I initially used the training and test data provided by Udacity for this. But due to the unbalabce of the training set, the initial few runs of the model were not accurate. I used several Augmentation techniques as described in the earlier to generate new training data nd make the training set more balanced. This lead to a better performance of the model at the end. My Final Validation accuracy came to 98.2%

My final model results were:
* training set accuracy of 99.2
* validation set accuracy of 99.2 
* test set accuracy of 91.0
 

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11] ![alt text][image12]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

##### Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      		| Priority road   									| 
| Keep right     			| Keep right 										|
| Turn left ahead					| Turn left ahead											|
| Speed limit (60km/h)     		| Speed limit (60km/h)					 				|
| Speed limit (30km/h)			| Speed limit (30km/h)      							|
| Road work			| Road work      							|


The model was able to correctly guess all 6 traffic signs, which gives an accuracy of 100%. 

#### Here is how the model predicted the correct value for the first image in the set. 

For the first image, Model predicted the Priority road sign with 100% accuracy. Below image shows what are the top 5 predictions for all the images in the 6 test images from the web.
![alt text][image13]



