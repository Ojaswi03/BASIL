import os
import numpy as np
import matplotlib.pyplot as plt
from data.cifar import load_cifar10
from data.mnist import load_mnist
from models import MNISTModel, CIFARModel
from basil import BasilNode, basil_ring_training_with_attack
from basil_plus import basil_plus_training
from trainer import evaluate
from acds import acds_share, apply_acds
import tensorflow as tf

def batchify(dataset, batch_size, dataset_choice='c'):
    data_loader = []
    X, y = [], []
    for i in range(len(dataset)):
        x_i, y_i = dataset[i]
        x_arr = np.array(x_i)

        if dataset_choice == 'c':
            # CIFAR-10 case: [3, 32, 32] → [32, 32, 3]
            if x_arr.ndim == 3 and x_arr.shape[0] == 3:
                x_arr = np.transpose(x_arr, (1, 2, 0))
        else:
            # MNIST case: [1, 28, 28] → [28, 28]
            if x_arr.ndim == 3 and x_arr.shape[0] == 1:
                x_arr = x_arr.squeeze(0)
            # In case somehow [28, 28, 1], just fix it too
            elif x_arr.ndim == 3 and x_arr.shape[-1] == 1:
                x_arr = x_arr[:, :, 0]

        X.append(x_arr)
        y.append(y_i)
        if len(X) == batch_size:
            data_loader.append((np.stack(X), np.array(y)))
            X, y = [], []
    if X:
        data_loader.append((np.stack(X), np.array(y)))
    return data_loader


def plot_accuracy_curve(accs, attack_type, S, output_dir="diagrams_M35"):
    os.makedirs(output_dir, exist_ok=True)
    folder_name = f"{attack_type}_S{S}"
    folder_path = os.path.join(output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    plt.figure()
    plt.plot(range(1, len(accs) + 1), accs, marker='o', label="Test Accuracy")
    plt.title(f"Basil Accuracy Curve - Attack: {attack_type}, S={S}")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(folder_path, f"{attack_type}_accuracy_curve.png"))
    plt.close()

def run_experiment(dataset_choice, num_nodes, S, rounds, epochs,
                   iid, use_acds, use_basil_plus, attack_type,
                   attacker_ids, batch_size=64, seed=42):

    np.random.seed(seed)

    if dataset_choice == 'c':
        user_data, test_set = load_cifar10(iid=iid, num_users=num_nodes, seed=seed)
        input_shape = (32, 32, 3)
        num_classes = 10
        ModelClass = CIFARModel
    else:
        user_data, test_set = load_mnist(iid=iid, num_users=num_nodes, seed=seed)
        input_shape = (28, 28)
        num_classes = 10
        ModelClass = MNISTModel

    test_loader = batchify(test_set, batch_size=batch_size, dataset_choice=dataset_choice)
    full_datasets = {i: user_data[i].dataset for i in range(num_nodes)}
    node_indices = {i: user_data[i].indices for i in range(num_nodes)}

    if not iid and use_acds:
        print("Applying ACDS for anonymous data sharing...")
        shared_indices_map = acds_share(list(range(num_nodes)), node_indices, alpha=0.05, H=5)
        for i in range(num_nodes):
            injected_data = apply_acds(full_datasets[i], shared_indices_map[i])
            full_datasets[i].data.extend([x[0] for x in injected_data])
            full_datasets[i].targets.extend([x[1] for x in injected_data])

    nodes = []
    for i in range(num_nodes):
        model = ModelClass(input_shape=input_shape, num_classes=num_classes)
        combined_dataset = [(full_datasets[i][idx][0], full_datasets[i][idx][1]) for idx in node_indices[i]]
        if not combined_dataset:
            raise ValueError(f"[ERROR] Node {i} has no assigned data. Check partitioning.")
        local_data = batchify(combined_dataset, batch_size=batch_size, dataset_choice=dataset_choice)
        node = BasilNode(node_id=i, model=model, data_loader=local_data, S=S)
        nodes.append(node)

    initial_params = nodes[0].model.get_params()
    for node in nodes:
        for _ in range(S):
            node.store_model(initial_params)

    if use_basil_plus:
        num_groups = int(num_nodes // S)
        trained_models = basil_plus_training(nodes, num_groups, S, rounds, epochs, test_loader)
        accs = [evaluate(model, test_loader) for model in trained_models]
    else:
        trained_models, accs = basil_ring_training_with_attack(nodes, rounds, epochs, test_loader, attack_type, attacker_ids)

    print("Experiment completed.")
    plot_accuracy_curve(accs, attack_type, S)

if __name__ == '__main__':
    dataset_choice = input("Select dataset (m for MNIST, c for CIFAR10): ").strip().lower()
    num_nodes = int(input("Enter number of nodes: "))
    S = int(input("Enter connectivity parameter S: "))
    rounds = int(input("Enter number of training rounds: "))
    epochs = int(input("Enter number of local epochs: "))
    iid_input = input("Use IID data distribution? (y/n): ").strip().lower()
    iid = iid_input == 'y'
    use_acds = False
    if not iid:
        acds_input = input("Apply ACDS sharing if non-IID? (y/n): ").strip().lower()
        use_acds = acds_input == 'y'
    basil_plus_input = input("Use Basil+ parallel training? (y/n): ").strip().lower()
    use_basil_plus = basil_plus_input == 'y'
    
    attacker_type = ['none', 'gaussian', 'sign_flip', 'hidden']
    
    for i in range(len(attacker_type)):
        if attacker_type[i] == 'none':
            attack_type = attacker_type[i]
            attacker_ids = []
        else:
            attack_type = attacker_type[i]
            attacker_ids = [2,5,7]

        print(f"\n Running attack type: {attack_type} with attacker IDs: {attacker_ids}\n")
        run_experiment(dataset_choice, num_nodes, S, rounds, epochs,
                       iid, use_acds, use_basil_plus, attack_type,
                       attacker_ids)
