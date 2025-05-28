import numpy as np
from copy import deepcopy
from trainer import local_update, evaluate

class BasilNode:
    def __init__(self, node_id, model, data_loader, S):
        self.node_id = node_id
        self.model = model
        self.data_loader = data_loader
        self.S = S
        self.stored_models = []
    
    
    def get_weights(self):
        return [w.numpy() for w in self.model.trainable_weights]
    
    def set_weights(self, weights):
        for var, w in zip(self.model.trainable_weights, weights):
            var.assign(w)

    def store_model(self, model_params):
        self.stored_models.append(deepcopy(model_params))
        if len(self.stored_models) > self.S:
            self.stored_models.pop(0)

    def select_model(self):
        losses = []
        for params in self.stored_models:
            self.model.set_params(params)
            total_loss = 0
            total = 0
            for X_batch, y_batch in self.data_loader:
                logits = self.model.forward(X_batch)
                total_loss += np.sum(np.maximum(0, logits - logits[np.arange(len(y_batch)), y_batch][:, np.newaxis] + 1) - 1)
                total += len(y_batch)
            losses.append(total_loss / total)
        best_idx = np.argmin(losses)
        return self.stored_models[best_idx]

    def update_model(self, epochs=1, lr=0.01):
        local_update(self.model, self.data_loader, epochs, lr)
        return deepcopy(self.model.get_params())


def basil_ring_training(nodes, rounds, epochs=1, test_loader=None):
    N = len(nodes)
    initial_model = deepcopy(nodes[0].model.get_params())

    for node in nodes:
        for _ in range(node.S):
            node.store_model(initial_model)

    test_accuracies = []

    for r in range(rounds):
        lr = 0.03 / (1 + 0.03 * r)
        print(f"\n[Round {r + 1}/{rounds}] Starting training step with lr={lr:.4f}...")

        for i in range(N):
            nodes[i].model.set_params(nodes[i].select_model())
            model_params = nodes[i].update_model(epochs, lr)

            for s in range(1, nodes[i].S + 1):
                neighbor_idx = (i + s) % N
                nodes[neighbor_idx].store_model(model_params)

        if test_loader is not None:
            round_acc = [evaluate(node.model, test_loader) for node in nodes]
            avg_acc = np.mean(round_acc)
            test_accuracies.append(avg_acc)
            print(f"[Round {r + 1}] Average Test Accuracy: {avg_acc:.4f}")

    return [node.model for node in nodes], test_accuracies
