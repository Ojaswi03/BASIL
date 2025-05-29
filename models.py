# import tensorflow as tf
# from tensorflow.keras import layers, models
# import numpy as np

# class MNISTModel:
#     def __init__(self, input_shape=(28, 28), num_classes=10):
#         self.model = models.Sequential([
#             layers.Flatten(input_shape=input_shape),
#             layers.Dense(100, activation='relu'),
#             layers.Dense(100, activation='relu'),
#             layers.Dense(num_classes)
#         ])

#     def get_params(self):
#         return self.model.get_weights()

#     def set_params(self, weights):
#         self.model.set_weights(weights)

#     @property
#     def trainable_weights(self):
#         return self.model.trainable_weights
#     @property
#     def trainable_variables(self):
#         return self.model.trainable_variables

#     def __call__(self, x, training=False):
#         return self.model(x, training=training)

# class CIFARModel:
#     def __init__(self, input_shape=(32, 32, 3), num_classes=10):
#         self.model = models.Sequential([
#             layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
#             layers.MaxPooling2D(pool_size=(2, 2)),
#             layers.Conv2D(32, (3, 3), activation='relu'),
#             layers.MaxPooling2D(pool_size=(2, 2)),
#             layers.Flatten(),
#             layers.Dense(100, activation='relu'),
#             layers.Dense(num_classes)
#         ])

#     def get_params(self):
#         return self.model.get_weights()

#     def set_params(self, weights):
#         self.model.set_weights(weights)

#     @property
#     def trainable_weights(self):
#         return self.model.trainable_weights
#     @property
#     def trainable_variables(self):
#         return self.model.trainable_variables  

#     def __call__(self, x, training=False):
#         return self.model(x, training=training)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class BaseModel:
    def __init__(self):
        self.model = None

    def build_model(self, input_shape, num_classes):
        raise NotImplementedError

    def train(self, train_loader, epochs=1, lr=0.01):
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        for epoch in range(epochs):
            for x, y in train_loader:
                if len(x.shape) == 3:
                    x = tf.expand_dims(x, axis=0)
                if x.shape[1] == 3 and x.shape[-1] != 3:
                    x = tf.transpose(x, perm=[0, 2, 3, 1])
                y = tf.convert_to_tensor(y)
                y = tf.reshape(y, [-1])
                with tf.GradientTape() as tape:
                    logits = self.model(x, training=True)
                    loss = loss_fn(y, logits)
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def evaluate(self, test_loader):
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        for x, y in test_loader:
            if len(x.shape) == 3:
                x = tf.expand_dims(x, axis=0)
            if x.shape[1] == 3 and x.shape[-1] != 3:
                x = tf.transpose(x, perm=[0, 2, 3, 1])
            y = tf.convert_to_tensor(y)
            y = tf.reshape(y, [-1])
            logits = self.model(x, training=False)
            acc_metric.update_state(y, logits)
        return acc_metric.result().numpy()


class MNISTModel(BaseModel):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        super().__init__()
        self.model = keras.Sequential([
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])


class CIFARModel(BaseModel):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super().__init__()
        self.model = keras.Sequential([
            keras.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes)
        ])
