# import numpy as np
# from copy import deepcopy
# from trainer import local_update, evaluate_batch_loss, evaluate

# class BasilNode:
#     def __init__(self, node_id, model, data_loader, S):
#         self.node_id = node_id
#         self.model = model
#         self.data_loader = data_loader
#         self.S = S
#         self.stored_models = []

#     def get_weights(self):
#         return [w.numpy() for w in self.model.trainable_weights]

#     def set_weights(self, weights):
#         for var, w in zip(self.model.trainable_weights, weights):
#             var.assign(w)

#     def store_model(self, model_params):
#         self.stored_models.append(deepcopy(model_params))
#         if len(self.stored_models) > self.S:
#             self.stored_models.pop(0)

#     def select_model(self):
#         losses = []
#         for params in self.stored_models:
#             self.model.set_params(params)
#             loss = evaluate_batch_loss(self.model, self.data_loader)
#             losses.append(loss)
#         best_idx = np.argmin(losses)
#         return self.stored_models[best_idx]

#     def update_model(self, epochs, lr):
#         local_update(self.model, self.data_loader, epochs, lr)
#         return deepcopy(self.model.get_params())

# def basil_ring_training(nodes, rounds, epochs, test_loader):
#     N = len(nodes)
#     initial_model = deepcopy(nodes[0].model.get_params())

#     for node in nodes:
#         for _ in range(node.S):
#             node.store_model(initial_model)

#     test_accuracies = []

#     for r in range(rounds):
#         lr = 0.03 / (1 + 0.03 * r)
#         print(f"\n[Round {r + 1}/{rounds}] Starting training step with lr={lr:.4f}...")

#         for i in range(N):
#             selected = nodes[i].select_model()
#             nodes[i].model.set_params(selected)
#             model_params = nodes[i].update_model(epochs, lr)

#             for s in range(1, nodes[i].S + 1):
#                 neighbor_idx = (i + s) % N
#                 nodes[neighbor_idx].store_model(model_params)

#         if test_loader is not None:
#             round_acc = [evaluate(node.model, test_loader) for node in nodes]
#             avg_acc = np.mean(round_acc)
#             test_accuracies.append(avg_acc)
#             print(f"[Round {r + 1}] Average Test Accuracy: {avg_acc:.4f}")

#     return [node.model for node in nodes], test_accuracies



import numpy as np
from attacks import apply_attack
from trainer import evaluate

class BasilNode:
    def __init__(self, node_id, model, data_loader, S):
        self.node_id = node_id
        self.model = model
        self.data_loader = data_loader
        self.S = S

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def local_train(self, epochs=1, lr=0.03):
        self.model.train(self.data_loader, epochs=epochs, lr=lr)

    def evaluate(self, test_loader):
        return evaluate(self.model, test_loader)

def basil_ring_training(nodes, rounds, epochs, test_loader, attack_type="none", attacker_ids=[]):
    num_nodes = len(nodes)
    accs = []
    for r in range(rounds):
        lr = 0.03 / (1 + 0.03 * r)  # Updated learning rate schedule
        print(f"[Round {r+1}/{rounds}] Starting training step with lr={lr:.4f}...")

        for node in nodes:
            node.local_train(epochs=epochs, lr=lr)

        for i in attacker_ids:
            if i < len(nodes):
                print(f"Injecting {attack_type} attack into node {i}")
                weights = nodes[i].get_weights()
                attacked_weights = apply_attack(weights, attack_type)
                nodes[i].set_weights(attacked_weights)

        new_weights = [None] * num_nodes
        for i in range(num_nodes):
            left = nodes[i]
            right = nodes[(i + 1) % num_nodes]
            lw = left.get_weights()
            rw = right.get_weights()
            agg = [(l + r) / 2 for l, r in zip(lw, rw)]
            new_weights[i] = agg

        for i in range(num_nodes):
            nodes[i].set_weights(new_weights[i])

        acc = np.mean([node.evaluate(test_loader) for node in nodes])
        print(f"[Round {r+1}] Average Test Accuracy: {acc:.4f}\n")
        accs.append(acc)

    return nodes, accs
