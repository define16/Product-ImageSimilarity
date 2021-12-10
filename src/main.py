import json
import os.path
import time
import urllib.request

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D

"""
https://twinw.tistory.com/252
"""

def read_data():
    with open(os.path.join("..", "data", "same_name.json")) as f:
        data = json.load(f)

    data_set = []
    before_url = ""
    for product_name, value in data.items():
        try:
            if before_url == value[0].get("대표이미지").split("/")[2]:
                time.sleep(1)
            data_set.append((product_name, value[0].get("대표이미지")))
            urllib.request.urlretrieve(value[0].get("대표이미지"),
                                       os.path.join("..", "data", "image", f"{product_name}.jpg"))
            before_url = value[0].get("대표이미지").split("/")[2]
        except:
            pass
        if len(data_set) == 27107:
            break


def get_more_data():
    data_set = []
    before_url = ""

    with open(os.path.join("..", "data", "same_name.json")) as f:
        data = json.load(f)

    for image in os.listdir(os.path.join("..", "data", "image"))[564:]:
        image_name = image.replace(".jpg", "")
        try:
            for i in range(1, 3, 1):
                if before_url == data.get(image_name)[i].get("대표이미지").split("/")[2]:
                    time.sleep(1)
                data_set.append((image_name, data.get(image_name)[i].get("대표이미지")))
                urllib.request.urlretrieve(data.get(image_name)[i].get("대표이미지"),
                                           os.path.join("..", "data", "image", f"{image_name}#####{i}.jpg"))
                before_url = data.get(image_name)[i].get("대표이미지").split("/")[2]
        except:
            pass
        if len(data_set) == 1000:
            break


def reprocessing():
    image_w = 28
    image_h = 28
    images = os.listdir(os.path.join("..", "data", "image"))
    X = []
    Y = []

    root_path = os.path.join("..", "data", "image")
    image_data = {}
    reduce_image_data = {}
    for image in images:
        if "#####" in image:
            original_image_name = image.split("#####")[0]
        else:
            original_image_name = image.replace(".jpg","")
        same_images = image_data.get(original_image_name, [])
        same_images.append(image)
        image_data[original_image_name] = same_images

    reduce_target_count = 0
    for image_name, value in image_data.items():
        if len(value) > 1:
            reduce_image_data[image_name] = value
        else:
            if reduce_target_count < 1000:
                reduce_image_data[image_name] = value
            reduce_target_count += 1

    image_data = reduce_image_data
    print(len(image_data))
    num_classes = len(image_data)
    for idex, image in enumerate(list(image_data.keys())):
        label = [0 for i in range(num_classes)]
        label[idex] = 1

        for filename in image_data.get(image):
            print(os.path.join(root_path,filename))
            img = cv2.imread(os.path.join(root_path,filename))
            img = cv2.resize(img, None, fx=image_w / img.shape[0], fy=image_h / img.shape[1])
            X.append(img / 256)
            Y.append(label)

    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    xy = (X_train, X_test, Y_train, Y_test)

    np.save("./img_data.npy", xy)


reprocessing()
#
# model = Sequential()
# model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu',
#                         input_shape=X_train.shape[1:]))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Convolution2D(64, 3, 3))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
# model.fit(X_train, Y_train, batch_size=32, nb_epoch=100)
#
# model.save('Gersang.h5')