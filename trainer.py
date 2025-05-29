import numpy as np
import tensorflow as tf

def local_update(model, data_loader, epochs, lr):
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for _ in range(epochs):
        for X_batch, y_batch in data_loader:
            X_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32)
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.int32)
            with tf.GradientTape() as tape:
                logits = model(X_batch, training=True)
                loss = loss_fn(y_batch, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return model

def evaluate(model, data_loader):
    correct = 0
    total = 0
    for X_batch, y_batch in data_loader:
        X_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        logits = model(X_batch, training=False)
        preds = tf.argmax(logits, axis=1).numpy()
        correct += np.sum(preds == y_batch)
        total += len(y_batch)
    return correct / total

def evaluate_batch_loss(model, data_loader):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    total_loss = 0.0
    total_samples = 0
    for X_batch, y_batch in data_loader:
        X_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.int32)
        logits = model(X_batch, training=False)
        batch_loss = loss_fn(y_batch, logits).numpy()
        total_loss += batch_loss * len(y_batch)
        total_samples += len(y_batch)
    return total_loss / total_samples if total_samples > 0 else float('inf')
