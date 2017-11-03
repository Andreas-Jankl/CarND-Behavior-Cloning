import tensorflow as tf
import numpy as np
import csv
import json
from keras.models import Sequential
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from scipy.misc import imread, imresize
import math

#Inspiration for this code came from:
#https://github.com/commaai/research/blob/master/train_steering_model.py
#https://github.com/dyelax/CarND-Behavioral-Cloning

# Flags to control script
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', 'data/driving_log_clean.csv', 'The path to the csv of training data.')
flags.DEFINE_integer('batch_size', 128, 'The minibatch size.')
flags.DEFINE_integer('num_epochs', 4, 'The number of epochs to train for.')
flags.DEFINE_integer('refine', 0, 'Flag controlling whether or not to fully train or just to refine model')
flags.DEFINE_string('refine_data_path', 'refine_data/driving_log_clean.csv', 'The path to the csv of training data for refining the model')
flags.DEFINE_float('lrate', 0.0001, 'The learning rate for training.')

# Path to the model weights file.
model_weights_path = 'model.h5'
model_weights_path_saved = 'model_saved.h5'

# Image dimensions
row, col, ch = 320, 160, 3 # camera format
row_new, col_new, ch_new = 32, 16, 1 # preprocessed image

	
# Read in the images into a numpy array
def read_imgs(img_paths):
    imgs = np.empty([len(img_paths), col, row, ch])

    for i, path in enumerate(img_paths):
        imgs[i] = imread(path)

    return imgs

	
# Resize the images to be 1/10 of the original dimensions in order to make training faster
def resize(imgs, shape=(row_new, col_new, 3)):
    """
    Resize images to shape.
    """
    height, width, channels = shape
    imgs_resized = np.empty([len(imgs), height, width, channels])
    for i, img in enumerate(imgs):
        imgs_resized[i] = imresize(img, shape)

    return imgs_resized

# Convert the images to grayscale in order to make training faster
def rgb2gray(imgs):
    """
    Convert images to grayscale.
    """
    return np.mean(imgs, axis=3, keepdims=True)

# Normalize the images to be between [-1,1]
def normalize(imgs):
    """
    Normalize images between [-1, 1].
    """
    return imgs / (255.0 / 2) - 1

# Functions which resizes, converts to gray and then normalizes the input image.
def preprocess(imgs):
    imgs_processed = resize(imgs)
    imgs_processed = rgb2gray(imgs_processed)
    imgs_processed = normalize(imgs_processed)

    return imgs_processed

# Since a lot of the curves are in one direction flipping that images horizontally shall lead to better results. as it avoids that the model is skewed in one direction while the model is still learning how to drive the curve.
def random_flip(imgs, angles):
    """
    Augment the data by randomly flipping some angles / images horizontally.
    """
    new_imgs = np.empty_like(imgs)
    new_angles = np.empty_like(angles)
    for i, (img, angle) in enumerate(zip(imgs, angles)):
        if np.random.choice(2):
            new_imgs[i] = np.fliplr(img)
            new_angles[i] = angle * -1
        else:
            new_imgs[i] = img
            new_angles[i] = angle

    return new_imgs, new_angles

# Function to call random flipping in case additional augmentations such as cropping or shifting the images are being introduced. Right now I don't use them.
def augment(imgs, angles):
    imgs_augmented, angles_augmented = random_flip(imgs, angles)

    return imgs_augmented, angles_augmented

	
# Generator in use. It randomly draws the images from the arrays reprocesses and augments them and then returns a batch. As requested for a generator it has an indefinite loop,.
def gen_batches(imgs, angles, batch_size):
    """
    Generates random batches of the input data.

    :param imgs: The input images.
    :param angles: The steering angles associated with each image.
    :param batch_size: The size of each minibatch.

    :yield: A tuple (images, angles), where both images and angles have batch_size elements.
    """
    num_elts = len(imgs)

    while True:
        indeces = np.random.choice(num_elts, batch_size)
        batch_imgs_raw, angles_raw = read_imgs(imgs[indeces]), angles[indeces].astype(float)
        batch_imgs, batch_angles = augment(preprocess(batch_imgs_raw), angles_raw)

        yield batch_imgs, batch_angles

def main(_):
    # Load Data
	if FLAGS.refine==0:
		with open(FLAGS.data_path, 'r') as f:
			reader = csv.reader(f)
			# data is a list of tuples (img path, steering angle)
			data = np.array([row for row in reader])
	
	if FLAGS.refine==1:
		with open(FLAGS.refine_data_path, 'r') as f:
			reader = csv.reader(f)
			# data is a list of tuples (img path, steering angle)
			data = np.array([row for row in reader])

    # Shuffle data and split train and validation data depending on whether one does full training or just refinement. Reason is that the number of images is different.
	np.random.shuffle(data)
	if FLAGS.refine==0:
		split_i = int(len(data) * 0.98)
	if FLAGS.refine==1:
		split_i = int(len(data) * 0.9)
	X_train, y_train = list(zip(*data[:split_i]))
	X_val, y_val = list(zip(*data[split_i:]))
	X_train, y_train = np.array(X_train), np.array(y_train)
	X_val, y_val = np.array(X_val), np.array(y_val)
	
	# Print out the Shapes of data sets so that one can see what is being fed while training
	print('Training data set',X_train.shape)
	print('Validation data set',X_val.shape)
	
    # Define Model
	model = Sequential()
	model.add(Conv2D(32, 3, 3, input_shape=(row_new, col_new, ch_new),border_mode='same', activation='relu'))
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
	
	# Compile Model. Using an Adam Optimizer with a lower learning rate than standard in order to get better results.
	# The loss is defined as mean squared error between the steering angle predicted and the steering angle from the data sets.
	model.compile(optimizer=Adam(lr=FLAGS.lrate), loss='mse')
	
	#In case of refinement training load pretrained model.
	if FLAGS.refine==1:
		model.load_weights(model_weights_path_saved, by_name=True)
		
    # Train the model with the help of a generator
	number_of_training_samples=int(math.ceil(len(X_train)/FLAGS.batch_size)*FLAGS.batch_size)
	number_of_validation_samples=int(math.ceil(len(X_val)/FLAGS.batch_size)*FLAGS.batch_size)
	history = model.fit_generator(gen_batches(X_train, y_train, FLAGS.batch_size),samples_per_epoch=number_of_training_samples,nb_epoch=FLAGS.num_epochs,validation_data=gen_batches(X_val, y_val, FLAGS.batch_size),nb_val_samples=number_of_validation_samples)
	
    # Save model structure as JSON file and the weights in .h5 file
	json_string = model.to_json()
	with open('model.json', 'w') as outfile:
		json.dump(json_string, outfile)
	model.save_weights(model_weights_path)
	

	
if __name__ == '__main__':
    tf.app.run()