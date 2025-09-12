import numpy as np
from copy import deepcopy
import tensorflow as tf

from attacks import apply_attack
from trainer import local_update, evaluate_batch_loss, evaluate


class BasilNode:
    """
    A single Basil node that maintains a memory of the last S models,
    selects the best-performing one on its local batch, updates locally,
    and forwards the updated weights to its next S neighbors.
    """
    def __init__(self, node_id, model, data_loader, S):
        self.node_id = node_id
        self.model = model
        self.data_loader = data_loader  # iterable of (X_batch, y_batch) pairs (NumPy or Tensors)
        self.S = int(S)
        self.stored_models = []  # list of parameter snapshots (lists of numpy arrays)

    # ----- Optional helpers (not strictly required by training loops) -----
    def get_weights(self):
        return [w.numpy() for w in self.model.trainable_weights]

    def set_weights(self, weights):
        for var, w in zip(self.model.trainable_weights, weights):
            var.assign(w)
    # ---------------------------------------------------------------------

    def store_model(self, model_params):
        """
        Keep a rolling window of size S of parameter snapshots.
        """
        self.stored_models.append(deepcopy(model_params))
        # keep at most S snapshots
        if len(self.stored_models) > self.S:
            self.stored_models.pop(0)

    def select_model(self):
        """
        Basil performance-based selection:
        evaluate each stored snapshot on a local batch; pick the one with the lowest loss.
        Falls back to the current model params if memory is empty.
        """
        if not self.stored_models:
            # No memory yet — return current model parameters
            return deepcopy(self.model.get_params())

        losses = []
        for params in self.stored_models:
            self.model.set_params(params)
            loss = evaluate_batch_loss(self.model, self.data_loader)
            losses.append(loss)

        best_idx = int(np.argmin(losses))
        return deepcopy(self.stored_models[best_idx])

    def update_model(self, epochs, lr):
        """
        Run local SGD and return the updated parameter snapshot.
        """
        local_update(self.model, self.data_loader, epochs, lr)
        return deepcopy(self.model.get_params())


def generate_adversarial_weights(model, data_loader, epsilon=0.1):
    """
    Craft adversarial-like weights close to benign ones using one batch FG-like step.
    This is used by the 'hidden' attack (blended with benign weights).
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    for X_batch, y_batch in data_loader:
        X_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.int32)
        with tf.GradientTape() as tape:
            logits = model(X_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_weights)

        # move a small step in gradient direction to get malicious weights
        new_weights = []
        for w, g in zip(model.get_weights(), grads):
            g_np = g.numpy()
            new_weights.append(w + epsilon * g_np)
        return new_weights

    # If no batches available, just return current params (no-op)
    return deepcopy(model.get_weights())


def _ensure_memory_initialized(nodes):
    """
    Make sure each node has S snapshots in memory before the first selection.
    Uses node 0's current params as the common initial snapshot.
    """
    if not nodes:
        return
    initial_params = deepcopy(nodes[0].model.get_params())
    for node in nodes:
        # If memory is empty or shorter than S, top it up
        if len(node.stored_models) < node.S:
            node.stored_models = []  # clear to avoid overfilling on re-entry
            for _ in range(node.S):
                node.store_model(initial_params)


def basil_ring_training(nodes, rounds, epochs, test_loader):
    """
    Standard Basil training without explicit attack injection.
    - Ensures memories are initialized.
    - Each node selects best snapshot, updates locally, and shares to S successors.
    - Tracks average test accuracy per round (if test_loader provided).
    """
    num_nodes = len(nodes)
    _ensure_memory_initialized(nodes)

    test_accuracies = []

    for r in range(int(rounds)):
        lr = 0.03 / (1 + 0.003 * r)   # decay 10x slower

        print(f"\n[Round {r + 1}/{rounds}] Starting training step with lr={lr:.4f}...")

        for i in range(num_nodes):
            # Select best snapshot by local batch loss
            selected = nodes[i].select_model()
            nodes[i].model.set_params(selected)

            # Local update
            model_params = nodes[i].update_model(epochs, lr)

            # Share to next S neighbors
            for s in range(1, nodes[i].S + 1):
                neighbor_idx = (i + s) % num_nodes
                nodes[neighbor_idx].store_model(model_params)

        # Evaluate average accuracy across nodes (optional)
        if test_loader is not None:
            round_acc = [evaluate(node.model, test_loader) for node in nodes]
            avg_acc = float(np.mean(round_acc))
            test_accuracies.append(avg_acc)
            print(f"[Round {r + 1}] Average Test Accuracy: {avg_acc:.4f}")

    return [node.model for node in nodes], test_accuracies


def basil_ring_training_with_attack(
    nodes,
    rounds,
    epochs,
    test_loader,
    attack_type,
    attacker_ids,
    hidden_start_round=20,
):
    """
    Basil training with optional Byzantine attacks.

    Args:
        nodes: list[BasilNode]
        rounds: int, number of global rounds
        epochs: int, local epochs per round
        test_loader: evaluation loader (list of (X_batch, y_batch)) or None
        attack_type: 'none' | 'gaussian' | 'sign_flip' | 'hidden'
        attacker_ids: list[int] indices of Byzantine nodes
        hidden_start_round: int, warm-up before hidden attack activates (default 20)
    """
    num_nodes = len(nodes)

    # Ensure memories are initialized (prevents argmin on empty list)
    _ensure_memory_initialized(nodes)

    test_accuracies = []

    for r in range(int(rounds)):
        lr = 0.03 / (1 + 0.003 * r)   # decay 10x slower

        print(f"\n[Round {r + 1}/{rounds}] Starting training step with lr={lr:.4f}...")

        for i in range(num_nodes):
            # Select and set the best snapshot
            selected = nodes[i].select_model()
            nodes[i].model.set_params(selected)

            # Local update
            model_params = nodes[i].update_model(epochs, lr)

            # Attack injection (if this node is Byzantine)
            if attack_type != 'none' and i in attacker_ids:
                benign_params = nodes[i].model.get_params()

                if attack_type == "hidden":
                    # Hidden attack is delayed until warm-up completes
                    if (r + 1) >= int(hidden_start_round):
                        adv_weights = generate_adversarial_weights(nodes[i].model, nodes[i].data_loader)
                        attacked = apply_attack(
                            benign_params,
                            attack_type="hidden",
                            malicious_weights=adv_weights,
                            blend_ratio=0.5,
                        )
                        nodes[i].model.set_params(attacked)
                        model_params = attacked  # share attacked params
                else:
                    attacked = apply_attack(benign_params, attack_type=attack_type)
                    nodes[i].model.set_params(attacked)
                    model_params = attacked

            # Share to next S neighbors
            for s in range(1, nodes[i].S + 1):
                neighbor_idx = (i + s) % num_nodes
                nodes[neighbor_idx].store_model(model_params)

        # Evaluate after each round (average across nodes)
        if test_loader is not None:
            round_acc = [evaluate(node.model, test_loader) for node in nodes]
            avg_acc = float(np.mean(round_acc))
            test_accuracies.append(avg_acc)
            print(f"[Round {r + 1}] Average Test Accuracy: {avg_acc:.4f}")

    return [node.model for node in nodes], test_accuracies
