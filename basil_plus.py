import numpy as np
from copy import deepcopy
from basil import basil_ring_training
from trainer import evaluate

def basil_plus_training(all_nodes, num_groups, S, rounds, epochs, test_loader):
    N = len(all_nodes)
    assert N % num_groups == 0, "Number of nodes must be divisible by number of groups"
    group_size = N // num_groups

    group_models = []
    for g in range(num_groups):
        print(f"\n=== Group {g + 1}/{num_groups} Ring Training ===")
        group_nodes = all_nodes[g * group_size:(g + 1) * group_size]
        trained_group_models, _ = basil_ring_training(group_nodes, rounds, epochs, test_loader)
        group_models.append(trained_group_models[-1].get_params())

    print("\n=== Aggregating Group Models (Robust Circular Aggregation) ===")
    # Circular aggregation: average received model with local one
    merged_model = group_models[0]
    for i in range(1, len(group_models)):
        merged_model = [((i * m1 + m2) / (i + 1)) for m1, m2 in zip(merged_model, group_models[i])]

    # Broadcast back the merged model
    for node in all_nodes:
        node.model.set_params(deepcopy(merged_model))

    if test_loader is not None:
        accs = [evaluate(node.model, test_loader) for node in all_nodes]
        print(f"\nFinal Averaged Accuracy: {np.mean(accs):.4f}")

    return [node.model for node in all_nodes]

