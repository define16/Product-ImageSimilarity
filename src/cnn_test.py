"""
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb
https://codetorial.net/tensorflow/convolutional_neural_network.html
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"X_train_shape: {x_train.shape}")
print(f"y_train_shape: {y_train.shape}")
print(f"X_test_shape: {x_test.shape}")
print(f"y_test_shape: {y_test.shape}")

# 2. 데이터 전처리하기
# mnist 데이터셋의 자료형을 출력해보면 numpy.ndarray 클래스임을 알 수 있습니다.
# NumPy의 reshape() 함수를 이용해서 적절한 형태로 변환하고,
# 0~255 사이의 값을 갖는 데이터를 0.0~1.0 사이의 값을 갖도록, 255.0으로 나눠줍니다.
train_images = x_train.reshape((60000, 28, 28, 1))
test_images = x_test.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0

# 3. 합성곱 신경망 구성하기
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
