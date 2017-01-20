import argparse
import base64
import json
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import matplotlib.image as mpimg
from scipy.misc.pilutil import imresize
from random import uniform
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf



# normalize image
def normalize(img):
    return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


# grayscale image
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# flip image
def flip_image(img):
    return cv2.flip(img, flipCode=1)


def change_brightness(img):
    change_pct = uniform(0.4, 1.2)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * change_pct
    img_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img_bright


def process_img(img, flip = False,  normalize_img=True, grayscale_img=True,
                         remove_top_bottom=True, remove_amount=0.125, resize=True, resize_percentage=.65,
                         brightness=True):
    # img = mpimg.imread('F:/CarNDstep/IMG/' + basename(file_name), 1)


    if remove_top_bottom:
        img = img[50:, :, :]


    if normalize_img:
        img = normalize(img)

    if flip:
        img = flip_image(img)

    if resize:
        img = imresize(img, resize_percentage, interp='bilinear', mode=None)

    if brightness:
        img = change_brightness(img)

    if grayscale_img:
        img = grayscale(img)
        img = img[..., np.newaxis]

    return img



sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

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
    # image_array = image_array[20:140, :]
    # image_array = imresize(image_array, .65, interp='bilinear', mode=None)


    image_array = process_img(image_array)

    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.15
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
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
          model = model_from_json(json.loads(jfile.read()))
        #
        # instead.
        # model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)