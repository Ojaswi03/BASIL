import numpy as np
import tensorflow as tf

# Global loss function (from Basil paper setup)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def local_update(model, data_loader, epochs, lr):
    """
    Perform local training update on a node's model using SGD.
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

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
    """
    Evaluate accuracy of model on the given dataset loader.
    """
    correct, total = 0, 0
    for X_batch, y_batch in data_loader:
        X_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        logits = model(X_batch, training=False)
        preds = tf.argmax(logits, axis=1).numpy()

        # ensure labels are NumPy array
        if hasattr(y_batch, "numpy"):
            y_true = y_batch.numpy()
        else:
            y_true = np.array(y_batch)

        correct += np.sum(preds == y_true)
        total += len(y_true)

    return correct / total if total > 0 else 0.0


def evaluate_batch_loss(model, data_loader):
    """
    Evaluate average cross-entropy loss of a model on a single batch.
    Used by Basil for selecting best snapshot from memory.
    """
    for X_batch, y_batch in data_loader:
        X_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.int32)
        logits = model(X_batch, training=False)
        loss = loss_fn(y_batch, logits).numpy()
        return loss
    return np.inf


def evaluate_all(nodes, test_loader):
    """
    Evaluate all nodes on the test set.
    Returns:
        avg_acc: average accuracy across nodes
        worst_acc: minimum (worst) accuracy among nodes
        acc_list: list of per-node accuracies
    """
    acc_list = [evaluate(node.model, test_loader) for node in nodes]
    avg_acc = float(np.mean(acc_list)) if acc_list else 0.0
    worst_acc = float(np.min(acc_list)) if acc_list else 0.0
    return avg_acc, worst_acc, acc_list
