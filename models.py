import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class MNISTModel:
    def __init__(self, input_shape=(28, 28), num_classes=10):
        self.model = models.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(100, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def get_params(self):
        return self.model.get_weights()

    def set_params(self, weights):
        self.model.set_weights(weights)

    def train(self, train_data, epochs):
        X, y = [], []
        for batch in train_data:
            X.append(batch[0])
            y.append(batch[1])
        X = tf.convert_to_tensor(np.vstack(X))
        y = tf.convert_to_tensor(np.hstack(y))
        self.model.fit(X, y, epochs=epochs, verbose=0)

    def evaluate(self, test_data):
        X, y = [], []
        for batch in test_data:
            X.append(batch[0])
            y.append(batch[1])
        X = tf.convert_to_tensor(np.vstack(X))
        y = tf.convert_to_tensor(np.hstack(y))
        return self.model.evaluate(X, y, verbose=0)[1]


class CIFARModel:
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        self.model = models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D(pool_size=(3, 3), strides=3),
            layers.Conv2D(64, (4, 4), activation='relu'),
            layers.MaxPooling2D(pool_size=(4, 4), strides=4),
            layers.Flatten(),
            layers.Dense(384, activation='relu'),
            layers.Dense(192, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def get_params(self):
        return self.model.get_weights()

    def set_params(self, weights):
        self.model.set_weights(weights)

    def train(self, train_data, epochs):
        X, y = [], []
        for batch in train_data:
            X.append(batch[0])
            y.append(batch[1])
        X = tf.convert_to_tensor(np.vstack(X))
        y = tf.convert_to_tensor(np.hstack(y))
        self.model.fit(X, y, epochs=epochs, verbose=0)

    def evaluate(self, test_data):
        X, y = [], []
        for batch in test_data:
            X.append(batch[0])
            y.append(batch[1])
        X = tf.convert_to_tensor(np.vstack(X))
        y = tf.convert_to_tensor(np.hstack(y))
        return self.model.evaluate(X, y, verbose=0)[1]
