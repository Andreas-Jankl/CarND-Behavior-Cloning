import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

from scipy.misc import imread, imresize

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

# Image dimensions
row, col, ch = 320, 160, 3 # camera format
row_new, col_new, ch_new = 32, 16, 1 # preprocessed image

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

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.

	# Preprocessing the image which means resize, convert to gray and then normalizes the input image.
    img_processed = preprocess(transformed_image_array)
	
	# Feed the network with the image in order to predict the steering angle
    steering_angle = float(model.predict(img_processed, batch_size=1))
   
   # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.1
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)