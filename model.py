import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import os, sys
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Input, ELU
from keras import initializations
from keras.models import load_model, model_from_json
from keras.layers.normalization import BatchNormalization
from sklearn.utils import shuffle
from keras import backend as K
import json
import gc

csv_path = 'driving_log.csv'  # my data (fantastic graphic mode)
csv_path1 = 'data/driving_log.csv'  # udacity data (fastest graphic mode)

center_db, left_db, right_db, steer_db = [], [], [], []
Rows, Cols = 64, 64
offset = 0.22

# read csv file
with open(csv_path1) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if float(row['steering']) != 0.0:
            center_db.append(row['center'])
            left_db.append(row['left'].strip())
            right_db.append(row['right'].strip())
            steer_db.append(float(row['steering']))
        else:
            prob = np.random.uniform()
            if prob <= 0.15:
                center_db.append(row['center'])
                left_db.append(row['left'].strip())
                right_db.append(row['right'].strip())
                steer_db.append(float(row['steering']))

# shuffle a dataset
center_db, left_db, right_db, steer_db = shuffle(center_db, left_db, right_db, steer_db)

# split train & valid data
img_train, img_valid, steer_train, steer_valid = train_test_split(center_db, steer_db, test_size=0.1, random_state=42)

plt.hist(steer_db, bins= 50, color= 'orange')
plt.xlabel('steering value')
plt.ylabel('counts')
# plt.show()

def select_img(center, left, right, steer, num, offsets=0.22):
    """
    randomly select among center, left, right images

    add ±0.22 to left, right steering angle.
    couldn't find exact left, right steering angle by using geometric method because we didn't have enough information.
    """
    rand = np.random.randint(3)

    if rand == 0:
        image, steering = cv2.imread(center[num]), steer[num]
    elif rand == 1:
        image, steering = cv2.imread(left[num]), steer[num] + offsets
    elif rand == 2:
        image, steering = cv2.imread(right[num]), steer[num] - offsets
    if abs(steering) > 1:
        steering = -1 if (steering < 0) else 1

    return image, steering

def valid_img(valid_image, valid_steer, num):
    """ using only center image for validation """
    steering = valid_steer[num]
    image = cv2.imread(valid_image[num])
    return image, steering

def crop_img(image):
    """ crop unnecessary parts """
    cropped_img = image[63:136, 0:319]
    resized_img = cv2.resize(cropped_img, (Cols, Rows), cv2.INTER_AREA)
    img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    return resized_img

def shift_img(image, steer):
    """
    randomly shift image horizontally
    add proper steering angle to each image
    """
    max_shift = 55
    max_ang = 0.14  # ang_per_pixel = 0.0025

    rows, cols, _ = image.shape

    random_x = np.random.randint(-max_shift, max_shift + 1)
    dst_steer = steer + (random_x / max_shift) * max_ang
    if abs(dst_steer) > 1:
        dst_steer = -1 if (dst_steer < 0) else 1

    mat = np.float32([[1, 0, random_x], [0, 1, 0]])
    dst_img = cv2.warpAffine(image, mat, (cols, rows))
    return dst_img, dst_steer

def brightness_img(image):
    """
    randomly change brightness by converting Y value
    """
    br_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    coin = np.random.randint(2)
    if coin == 0:
        random_bright = 0.2 + np.random.uniform(0.2, 0.6)
        br_img[:, :, 2] = br_img[:, :, 2] * random_bright
    br_img = cv2.cvtColor(br_img, cv2.COLOR_HSV2RGB)
    return br_img

def generate_shadow(image, min_alpha=0.5, max_alpha = 0.75):
    """generate random shadow in random region"""

    top_x, bottom_x = np.random.randint(0, Cols, 2)
    coin = np.random.randint(2)
    rows, cols, _ = image.shape
    shadow_img = image.copy()
    if coin == 0:
        rand = np.random.randint(2)
        vertices = np.array([[(50, 65), (45, 0), (145, 0), (150, 65)]], dtype=np.int32)
        if rand == 0:
            vertices = np.array([[top_x, 0], [0, 0], [0, rows], [bottom_x, rows]], dtype=np.int32)
        elif rand == 1:
            vertices = np.array([[top_x, 0], [cols, 0], [cols, rows], [bottom_x, rows]], dtype=np.int32)
        mask = image.copy()
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (0,) * channel_count
        cv2.fillPoly(mask, [vertices], ignore_mask_color)
        rand_alpha = np.random.uniform(min_alpha, max_alpha)
        cv2.addWeighted(mask, rand_alpha, image, 1 - rand_alpha, 0., shadow_img)

    return shadow_img

def flip_img(image, steering):
    """ randomly flip image to gain right turn data (track1 is biaed in left turn) """
    flip_image = image.copy()
    flip_steering = steering
    num = np.random.randint(2)
    if num == 0:
        flip_image, flip_steering = cv2.flip(image, 1), -steering
    return flip_image, flip_steering

def network_model():
    """
    designed with 4 convolutional layer & 3 fully connected layer
    weight init : glorot_uniform
    activation func : relu
    pooling : maxpooling
    used dropout
    """

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(Rows, Cols, 3)))
    model.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2), activation='relu', name='Conv1'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation='relu', name='Conv2'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(1, 1), activation='relu', name='Conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    #model.add(BatchNormalization())
    model.add(Convolution2D(128, 2, 2, border_mode='same', subsample=(1, 1), activation='relu', name='Conv4'))
    #model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', name='FC1'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', name='FC2'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', name='FC3'))
    model.add(Dense(1))
    model.summary()
    return model

def generate_train(center, left, right, steer):
    """
    data augmentation
    transformed image & crop
    """

    num = np.random.randint(0, len(steer))
    # to avoid bias in straight angle
    #bal = True
    #while bal:
    #    num = np.random.randint(0, len(steer))
    #    check_steer = steer[num]
    #    if check_steer == 0:
    #        rand = np.random.uniform()
    #        if rand <= 0.25:
    #            bal = False
    #    else:
    #        bal = False

    image, steering = select_img(center, left, right, steer, num, offset)

    image, steering = shift_img(image, steering)
    image, steering = flip_img(image, steering)
    image = brightness_img(image)
    # image = generate_shadow(image)
    image = crop_img(image)
    return image, steering

def generate_valid(img_valid, steer_valid):
    """ generate validation set """
    img_set = np.zeros((len(img_valid), Rows, Cols, 3))
    steer_set = np.zeros(len(steer_valid))

    for i in range(len(img_valid)):
        img, steer = valid_img(img_valid, steer_valid, i)
        img_set[i] = crop_img(img)

        steer_set[i] = steer
    return img_set, steer_set

def generate_train_batch(center, left, right, steering, batch_size):
    """ compose training batch set """
    image_set = np.zeros((batch_size, Rows, Cols, 3))
    steering_set = np.zeros(batch_size)

    while 1:
        for i in range(batch_size):
            img, steer = generate_train(center, left, right, steering)
            image_set[i] = img
            steering_set[i] = steer
        yield image_set, steering_set

batch_size = 256
epoch = 10

train_generator = generate_train_batch(center_db, left_db, right_db, steer_db, batch_size)
image_val, steer_val = generate_valid(img_valid, steer_valid)

model = network_model()

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mse')

model_json = 'model.json'
model_weights = 'model.h5'

history = model.fit_generator(train_generator, samples_per_epoch=20480, nb_epoch=epoch,
                              validation_data=(image_val, steer_val), verbose=1)

json_string = model.to_json()

try:
    os.remove(model_json)
    os.remove(model_weights)
except OSError:
    pass

with open(model_json, 'w') as jfile:
    json.dump(json_string, jfile)
model.save_weights(model_weights)

# to avoid " 'NoneType' object has no attribute 'TF_DeleteStatus' " error
gc.collect()
K.clear_session()
