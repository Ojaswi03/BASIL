# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from data.cifar import load_cifar10
# from data.mnist import load_mnist
# from models import MNISTModel, CIFARModel
# from basil import BasilNode, basil_ring_training
# from basil_plus import basil_plus_training
# from trainer import evaluate
# from acds import acds_share, apply_acds


# def batchify(dataset, batch_size):
#     data_loader = []
#     X, y = [], []
#     for i in range(len(dataset)):
#         x_i, y_i = dataset[i]
#         x_arr = np.array(x_i)
#         if x_arr.ndim == 3 and x_arr.shape[0] == 3:
#             x_arr = np.transpose(x_arr, (1, 2, 0))
#         X.append(x_arr)
#         y.append(y_i)
#         if len(X) == batch_size:
#             data_loader.append((np.stack(X), np.array(y)))
#             X, y = [], []
#     if X:
#         data_loader.append((np.stack(X), np.array(y)))
#     return data_loader


# def plot_accuracy_curve(accs, attack_type, S, output_dir="diagrams"):
#     os.makedirs(output_dir, exist_ok=True)
#     folder_name = f"{attack_type}_S{S}"
#     folder_path = os.path.join(output_dir, folder_name)
#     os.makedirs(folder_path, exist_ok=True)
#     plt.figure()
#     plt.plot(range(10, 10 + 10 * len(accs), 10), accs, marker='o', label="Test Accuracy")
#     plt.title(f"Basil Accuracy Curve\nAttack: {attack_type}, S={S}")
#     plt.xlabel("Round")
#     plt.ylabel("Accuracy")
#     plt.grid(True)
#     plt.legend()
#     cleaned_name = "clean" if attack_type == "none" else attack_type
#     filename = f"{cleaned_name}_accuracy_curve.png"
#     save_path = os.path.join(folder_path, filename)
#     plt.savefig(save_path)
#     plt.close()
#     print(f"✅ Saved plot to: {save_path}")


# def run_experiment(dataset_choice, num_nodes, S, rounds, epochs,
#                    iid, use_acds, use_basil_plus, attack_type,
#                    attacker_ids, batch_size=64, seed=42):
#     from attacks import apply_attack
#     np.random.seed(seed)
#     if dataset_choice == 'c':
#         user_data, test_set = load_cifar10(iid=iid, num_users=num_nodes, seed=seed)
#         input_shape = (32, 32, 3)
#         num_classes = 10
#         ModelClass = CIFARModel
#     else:
#         user_data, test_set = load_mnist(iid=iid, num_users=num_nodes, seed=seed)
#         input_shape = (28, 28)
#         num_classes = 10
#         ModelClass = MNISTModel
#     test_loader = batchify(test_set, batch_size=batch_size)
#     full_datasets = {i: user_data[i].dataset for i in range(num_nodes)}
#     node_indices = {i: user_data[i].indices for i in range(num_nodes)}
#     if not iid and use_acds:
#         print("Applying ACDS for anonymous data sharing...")
#         shared_indices_map = acds_share(list(range(num_nodes)), node_indices, alpha=0.05, H=5)
#         for i in range(num_nodes):
#             injected_data = apply_acds(full_datasets[i], shared_indices_map[i])
#             full_datasets[i].data.extend([x[0] for x in injected_data])
#             full_datasets[i].targets.extend([x[1] for x in injected_data])
#     nodes = []
#     for i in range(num_nodes):
#         model = ModelClass(input_shape=input_shape, num_classes=num_classes)
#         combined_dataset = [(full_datasets[i][idx][0], full_datasets[i][idx][1]) for idx in node_indices[i]]
#         local_data = batchify(combined_dataset, batch_size=batch_size)
#         node = BasilNode(node_id=i, model=model, data_loader=local_data, S=S)
#         nodes.append(node)
#     if use_basil_plus:
#         num_groups = int(num_nodes // S)
#         trained_models = basil_plus_training(nodes, num_groups, S, rounds, epochs, test_loader)
#         accs = [evaluate(model, test_loader) for model in trained_models]
#     else:
#         trained_models, accs = basil_ring_training(nodes, rounds, epochs, test_loader, attack_type, attacker_ids)
#     print("Experiment completed.")
#     plot_accuracy_curve(accs, attack_type, S)


# if __name__ == '__main__':
#     dataset_choice = input("Select dataset (m for MNIST, c for CIFAR10): ").strip().lower()
#     num_nodes = int(input("Enter number of nodes: "))
#     S = int(input("Enter connectivity parameter S: "))
#     rounds = int(input("Enter number of training rounds: "))
#     epochs = int(input("Enter number of local epochs: "))
#     iid_input = input("Use IID data distribution? (y/n): ").strip().lower()
#     iid = iid_input == 'y'
#     use_acds = False
#     if not iid:
#         acds_input = input("Apply ACDS sharing if non-IID? (y/n): ").strip().lower()
#         use_acds = acds_input == 'y'
#     basil_plus_input = input("Use Basil+ parallel training? (y/n): ").strip().lower()
#     use_basil_plus = basil_plus_input == 'y'
#     # attacker_type = ['none', 'gaussian', 'sign_flip', 'hidden']
#     attacker_type = ['gaussian']
#     attacker_ids = [2, 5, 7]

#     for i in range(len(attacker_type)):
#         attack_type = attacker_type[i]
#         if i == 0 and attack_type == 'none':
#             attacker_ids = []

#         print(f"====================================================================================================================")
#         print(f"Running experiment with dataset={dataset_choice}, num_nodes={num_nodes}, S={S}, "
#               f"rounds={rounds}, epochs={epochs}, iid={iid}, use_acds={use_acds}, "
#               f"use_basil_plus={use_basil_plus}, attack_type={attack_type}, attacker_ids={attacker_ids}")
#         print(f"====================================================================================================================")
#         run_experiment(dataset_choice, num_nodes, S, rounds, epochs,
#                        iid, use_acds, use_basil_plus, attack_type, attacker_ids)



import numpy as np
import matplotlib.pyplot as plt
import os
from data.cifar import load_cifar10
from data.mnist import load_mnist
from models import MNISTModel, CIFARModel
from basil import BasilNode, basil_ring_training
from acds import acds_share, apply_acds
from attacks import apply_attack

def run_experiment(dataset_choice, num_nodes, S, rounds, epochs, iid, use_acds, use_basil_plus, attack_type, attacker_ids):
    # Load dataset
    if dataset_choice == 'm':
        input_shape = (28, 28, 1)
        num_classes = 10
        ModelClass = MNISTModel
        load_data = load_mnist
    else:
        input_shape = (32, 32, 3)
        num_classes = 10
        ModelClass = CIFARModel
        load_data = load_cifar10

    train_loaders, test_loader = load_data(num_users=num_nodes, iid=iid, seed=42)


    # Apply ACDS data sharing if enabled
    if use_acds:
        print("📤 Applying ACDS sharing...")
        group_nodes = list(range(num_nodes))
        group_data = {i: train_loaders[i].dataset for i in range(num_nodes)}
        shared_indices = acds_share(group_nodes, group_data)
        for i in range(num_nodes):
            injected_data = apply_acds(group_data[i], shared_indices[i])
            train_loaders[i].dataset.data = injected_data  # Replace data with shared

    # Initialize nodes
    nodes = []
    for i in range(num_nodes):
        model = ModelClass(input_shape=input_shape, num_classes=num_classes)
        node = BasilNode(i, model, train_loaders[i], S)
        nodes.append(node)

    # Train using Basil Ring
    trained_models, accs = basil_ring_training(
        nodes, rounds, epochs, test_loader,
        attack_type=attack_type,
        attacker_ids=attacker_ids
    )

    # Save plot
    dir_name = f"diagrams/{attack_type}_S{S}"
    os.makedirs(dir_name, exist_ok=True)
    plt.plot(range(1, rounds + 1), accs)
    plt.xlabel("Rounds")
    plt.ylabel("Average Accuracy")
    plt.title(f"Basil Accuracy Curve ({attack_type})")
    plt.grid(True)
    save_path = f"{dir_name}/{attack_type}_accuracy_curve.png"
    plt.savefig(save_path)
    print(f"✅ Saved plot to: {save_path}")



if __name__ == "__main__":
    print("🔁 Starting Basil Experiments...")
    dataset_choice = input("Select dataset (m for MNIST, c for CIFAR10): ").lower()
    num_nodes = int(input("Enter number of nodes: "))
    S = int(input("Enter connectivity parameter S: "))
    rounds = int(input("Enter number of training rounds: "))
    epochs = int(input("Enter number of local epochs: "))
    iid_choice = input("Use IID data distribution? (y/n): ").lower()
    use_acds = iid_choice != 'y'
    use_basil_plus = input("Use Basil+ parallel training? (y/n): ").lower() == 'y'

    # Attack configs (can modify for sweeps)
    attack_type = "gaussian"  # Change to "sign_flip" or "none" as needed
    attacker_ids = [2, 5, 7]

    run_experiment(
        dataset_choice, num_nodes, S, rounds, epochs,
        iid=not use_acds, use_acds=use_acds, use_basil_plus=use_basil_plus,
        attack_type=attack_type, attacker_ids=attacker_ids
    )

    print("✅ All experiments completed.")
