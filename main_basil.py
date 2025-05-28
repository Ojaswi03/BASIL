import os
import numpy as np
from data.mnist import load_mnist
from data.cifar import load_cifar10
from models import MNISTModel, CIFARModel
from basil import BasilNode, basil_ring_training
from basil_plus import basil_plus_training
from trainer import evaluate
from acds import acds_share, apply_acds
from attacks import apply_attack
import matplotlib.pyplot as plt

# ---- Wrap Test Set ----
def batchify(dataset, batch_size=64):
    data_loader = []
    X, y = [], []
    for i in range(len(dataset)):
        x_i, y_i = dataset[i]
        X.append(np.array(x_i).squeeze())
        y.append(y_i)
        if len(X) == batch_size:
            data_loader.append((np.stack(X), np.array(y)))
            X, y = [], []
    if X:
        data_loader.append((np.stack(X), np.array(y)))
    return data_loader

# ---- Main Execution ----
if __name__ == '__main__':
    print("Welcome to Basil Trainer 🧪")
    dataset_choice = input("Select dataset - MNIST or CIFAR10 (m/c): ").strip().lower()
    num_nodes = int(input("Enter number of nodes: "))
    S = int(input("Enter connectivity parameter S: "))
    rounds = int(input("Enter number of training rounds: "))
    epochs = int(input("Enter number of local epochs: "))
    iid_input = input("Use IID data distribution? (y/n): ").strip().lower()
    iid = iid_input == 'y'
    acds_toggle = input("Apply ACDS sharing if non-IID? (y/n): ").strip().lower() == 'y'
    basil_plus_toggle = input("Use Basil+ parallel training? (y/n): ").strip().lower() == 'y'
    attack_types = input("Select attack types (comma-separated: none,gaussian,sign_flip,hidden): ").strip().lower().split(',')
    attacker_ids = input("Enter comma-separated attacker node IDs (or leave blank): ").strip()

    np.random.seed(42)

    if dataset_choice == 'c':
        user_data, test_set = load_cifar10(iid=iid, num_users=num_nodes)
        input_shape = (32, 32, 3)
        num_classes = 10
        ModelClass = CIFARModel
        diagram_folder = 'cifar_diagram'
    else:
        user_data, test_set = load_mnist(iid=iid, num_users=num_nodes)
        input_shape = (28, 28)
        num_classes = 10
        ModelClass = MNISTModel
        diagram_folder = 'mnist_diagram'

    os.makedirs(diagram_folder, exist_ok=True)

    test_loader = batchify(test_set)
    full_datasets = {i: user_data[i].dataset for i in range(num_nodes)}
    node_indices = {i: user_data[i].indices for i in range(num_nodes)}

    if not iid and acds_toggle:
        print("Applying ACDS for anonymous data sharing...")
        shared_indices_map = acds_share(list(range(num_nodes)), node_indices, alpha=0.05, H=5)
        for i in range(num_nodes):
            injected_data = apply_acds(full_datasets[i], shared_indices_map[i])
            for item in injected_data:
                full_datasets[i].data.append(item[0])
                full_datasets[i].targets.append(item[1])

    attacker_ids = [int(a.strip()) for a in attacker_ids.split(',')] if attacker_ids else []

    for atk in attack_types:
        nodes = []
        for i in range(num_nodes):
            model = ModelClass(input_shape=input_shape, num_classes=num_classes)
            combined_dataset = [(full_datasets[i][idx][0], full_datasets[i][idx][1]) for idx in node_indices[i]]
            local_data = batchify(combined_dataset)
            node = BasilNode(node_id=i, model=model, data_loader=local_data, S=S)
            if i in attacker_ids and atk != 'none':
                print(f"Injecting {atk} attack into node {i}")
                weights = node.get_weights()
                attacked = apply_attack(weights, attack_type=atk)
                node.set_weights(attacked)
            nodes.append(node)

        if basil_plus_toggle:
            G = int(input("Enter number of groups (G): "))
            trained_models, test_accuracies = basil_plus_training(nodes, G=G, S=S, rounds=rounds, epochs=epochs, test_loader=test_loader)
        else:
            trained_models, test_accuracies = basil_ring_training(nodes, rounds=rounds, epochs=epochs, test_loader=test_loader)

        accuracies = [evaluate(model, test_loader) for model in trained_models]
        avg_acc = np.mean(accuracies)
        print(f"[Attack: {atk}] Test Accuracies of each node:", accuracies)
        print(f"[Attack: {atk}] Average Accuracy:", avg_acc)

        # --- Plotting Round-by-Round Accuracy ---
        if test_accuracies:
            plt.plot(range(1, rounds + 1), test_accuracies, marker='o', label=atk)
            plt.title("Test Accuracy per Round")
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.grid(True)
            fname = f"{dataset_choice}_S{S}_R{rounds}_E{epochs}_ATK{atk}_ACC{avg_acc:.2f}.png"
            fpath = os.path.join(diagram_folder, fname)
            plt.savefig(fpath)
            print(f"Plot saved to {fpath}")
            plt.clf()
