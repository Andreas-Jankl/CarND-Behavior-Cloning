# Question 1

-The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.

> **Prerequesites**
>
>-  A model needs to use convolutional layers since it needs to do image processing
> - The output layer needs to have size 1 meaning it puts out the steering angle.
> - The loss for learning needs to be defined as as mean squared error between the steering angle predicted and the steering angle from the data sets.


> **I have tried 3 approaches for this problem**

> - Transfer learning using a vgg16 network as basis
> - A model inspired by comma.ai using 3 convolution and 1 dense layer
> - A model incorporating the learnings from the first two tries plus learnings from the slack and confluence comunity posts

**Try 1**
For try 1 the base layers of a vgg16 network are being used with which bottleneck features are being calculated. Only the convolutional layers with their imagenet trained weights are being used for the bottleneck data calculation. With the bottleneck data a number of new layers = top model are being trained which shall reside on top of the vgg network. The output of the top layers is a single output with a linear activation function which shall be the steering angle which shall be set

> **End result:** This model was found to need a very very long time to calculate the bootleneck data. I ended up giving up for another architecture since it was too complex for the task.

**Try 2**
The model was inspired by this: https://github.com/commaai/research/blob/master/train_steering_model.py However even with a lot of data I could not make the model run through the hard turns. 

I did try the following data with different sizes of the data sets and hyper parameters:
>- Using only the center images + recovery recording
>- Using the left/right images adding and subtracting 0.25 from the steering angle
>- I did not do any preprocessing or augmenting of the data

Finally I concluded the following:

> **End result:** The model is underfitting the data significantly because there were to little Dense layers. In fact there are 3 convolutional layers and only 1 Dense layer. Howeve the fact that I did not preprocess the data and therefore the model had a lot of parameters to be trained lead to the poorer results.

**Try 3**
With the learnings + hints from the students posting their models and their hints in confluence and slack particularily this https://github.com/dyelax/CarND-Behavioral-Cloning I came up with the last model which has 4 convolutational layers and 4 dense layers.

> **End result:** This model finally is able to fit the data. The reason for it being able to fit the data is in the increased number of layers compared to try 2. Additionally I am now preprocessing and augmenting the data (See below)


# Question 3

The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

Code snippet:
```sh
	model = Sequential()
	model.add(Conv2D(32, 3, 3,input_shape=(row_new, col_new, ch_new),border_mode='same', activation='relu'))
	model.add(Conv2D(64, 3, 3,border_mode='same', activation='relu'))
	model.add(Dropout(.5))
	model.add(Conv2D(128, 3, 3,border_mode='same', activation='relu'))
	model.add(Conv2D(256, 3, 3,border_mode='same', activation='relu'))
	model.add(Dropout(.5))
	model.add(Flatten())
	model.add(Dense(1024,activation='relu'))
	model.add(Dense(512,activation='relu'))
	model.add(Dense(128,activation='relu'))
	model.add(Dense(1, name='output', activation='tanh'))
```
	
>- 4 convolutational layers are being used which detect increasingly complex structures in the images
>- Twice dropout is being applied to avoid overfitting
>- 4 Dense layers with decreasing size are being added
>- All layers are using a relu activation except for the last one which uses a tanh activation which keeps the steering angle in [-1,1]

# Question 3

The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset should be included.

**The dataset was derived as follows:**
>- 6 rounds of center line driving was recorded
>- 4 round of repeatedly placing the vehicle at the left/side side of the road without recording and then recording the part where I steer the vehicle back in the middle was recorded
>- Everything was recorded on track 1
>- Only the center images where being taken
>- I ended up having approx 20k images
>- Unfortunately I used the keyboard only controls which makes up very weird steering angles and the 10Hz Simulator

**Traing was done as follows:**
>- I trained for 4 epochs with an adam optimizer and a mean squared error as loss definition. I tried different number of epochs. 4 was chosen as every number below ended in being underfitting and every higher number ended in overfitting
>- I used the a batch size of 128. Lower numbers tended to overfit the model
>- I used a learning rate of 0.0001 which is lower than the standard adam optimizer learning rate which lead to better results
>- I did use a generator as required in order to avoid the need to keep the data in memory.
>- The data was shuffled and split into training and validation data set. Testing was done with the simulator itself so not testing dataset was derived.
>- In the batch generation the data is being preprocess as follows:
```flow
st=>start: Start
op1=>operation: Resize
op2=>operation: Convert to grayscale
op3=>operation: Normalize [-1,1]

st->op1->op2->op3->e
```
>- The batch generation itself works as follows:
```flow
st=>start: Start
op1=>operation: Get random indices
op2=>operation: Read in images with given indices
op3=>operation: Augment data

st->op1->op2->op3->e
```
>- The augmentation does random horizontal flipping of the images. I do this because the track is very scewed with curves mostly in one direction. In order to avoid a scewed model this technique is being used.
>- Resizing the images and using gray scale only was being chosen at it reduced the paramters and thus training time significantly while the model was still able to fit the data.
>- I ended up training the model and then once I had a generally running model I did refine the working model with data at the places where the car fell of track. Whether the model.py script does full training or refining is being controlled by flag in the code.