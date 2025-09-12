import tensorflow as tf
from tensorflow.keras import layers, models


class MNISTModel:
    """
    MNIST model as in Basil paper:
    3 fully connected layers: 784x100 -> 100x100 -> 100x10
    """
    def __init__(self, input_shape=(28, 28), num_classes=10):
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(num_classes)  # logits, no activation
        ])

    def get_params(self):
        return self.model.get_weights()

    def set_params(self, weights):
        self.model.set_weights(weights)

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def __call__(self, x, training=False):
        return self.model(x, training=training)


class CIFARModel:
    """
    CIFAR model as in Basil paper:
    2 Conv layers + 3 FC layers
    conv1: 3x16x3x3, conv2: 16x64x4x4
    fc1: 64x384, fc2: 384x192, fc3: 192x10
    """
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(3, 3), strides=3),
            layers.Conv2D(64, (4, 4), activation='relu'),
            layers.MaxPooling2D(pool_size=(4, 4), strides=4),
            layers.Flatten(),
            layers.Dense(384, activation='relu'),
            layers.Dense(192, activation='relu'),
            layers.Dense(num_classes)  # logits, no activation
        ])

    def get_params(self):
        return self.model.get_weights()

    def set_params(self, weights):
        self.model.set_weights(weights)

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def __call__(self, x, training=False):
        return self.model(x, training=training)
