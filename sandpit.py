import pandas as pd
import cv2
import numpy as np
from random import shuffle
from os.path import basename


# this is because windows loves backslashes while everyone else is on forward
def fokken_windows(path):
    s = path.split('\\')
    return 'IMG/' + s[-1]


# variables
batch_size = 64
image_dimension_x = 320
image_dimension_y = 160
# image_dimension_depth = 1
learning_rate = 0.01
nb_epoch = 10

# options
use_all_three_images = False
grayscale_images = True
normalize_images = True

# and on the 8th day (variables that follow from inputs)

if grayscale_images:
    color_depth = 1
else:
    color_depth = 3
if use_all_three_images:
    image_depth = 3
else:
    image_depth = 1
image_shape = (image_dimension_x, image_dimension_y, color_depth * image_depth)

########################
# Additional functions #
########################


# normalize image
def normalize(img):
    return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


# grayscale image
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# image read and process function
def read_and_process_img(file_name, normalize_img=normalize_images, grayscale_img=grayscale_images):
    file_name = 'IMG/' + basename(file_name)
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    if normalize_img:
        img = normalize(img)
    if grayscale_img:
        img = grayscale(img)
    return img


def img_generator(images):
    ii = 0
    while 1:
        images_out = np.ndarray(shape=(batch_size, image_dimension_x, image_dimension_y, color_depth * image_depth),
                                dtype=float)  # n*x*y*RGG
        angles_out = np.ndarray(shape=batch_size, dtype=float)
        # print(images_out.shape)
        for j in range(batch_size):
            if ii >= len(images):
                shuffle(images)
                ii = 0
            # print(fokken_windows(images[ii][0]))
            print(images[ii][0])
            centre = read_and_process_img(images[ii][0]).reshape(image_dimension_x, image_dimension_y,  color_depth * image_depth)
            print(centre.shape)
            left = read_and_process_img(images[ii][1]).reshape(image_dimension_x, image_dimension_y,  color_depth * image_depth)
            print(left.shape)
            right = read_and_process_img(images[ii][2]).reshape(image_dimension_x, image_dimension_y,  color_depth * image_depth)
            print(right.shape)
            print(centre.shape)
            angle = images[ii][3]
            if use_all_three_images:
                images_out[ii] = np.dstack((centre, left, right))
            else:
                images_out[ii] = centre
            angles_out[ii] = angle
            angles_out = angles_out.reshape(batch_size, 1)
            ii += 1
        return images_out, angles_out






with open('driving_log.csv') as f:
    logs = pd.read_csv(f, header=None)
    logs = logs.ix[:,[0, 1, 2, 3]]

for i in range(len(logs)):
    # print(logs[0][i])
    img = read_and_process_img(logs[0][i])
    print(img.shape)