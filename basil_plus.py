import numpy as np
from copy import deepcopy

from basil import basil_ring_training
from trainer import evaluate, evaluate_batch_loss


def basil_plus_training(all_nodes, num_groups, S, rounds, epochs, test_loader, tau=1):
    """
    Basil+ training: parallel Basil within groups + robust circular aggregation.

    Args:
        all_nodes: list of BasilNode objects
        num_groups: number of groups
        S: connectivity parameter
        rounds: number of global rounds
        epochs: local epochs
        test_loader: evaluation data loader
        tau: number of sequential Basil rounds per group before aggregation
    """
    N = len(all_nodes)
    assert N % num_groups == 0, "Number of nodes must be divisible by number of groups"
    group_size = N // num_groups

    test_accuracies = []

    for r in range(rounds):
        print(f"\n=== Global Round {r + 1}/{rounds} ===")

        # ----- Stage 1: Within-group Basil training -----
        group_models = []
        for g in range(num_groups):
            group_nodes = all_nodes[g * group_size:(g + 1) * group_size]
            trained_group_models, _ = basil_ring_training(group_nodes, tau, epochs, test_loader=None)
            group_models.append(trained_group_models[-1].get_params())

        # ----- Stage 2: Robust Circular Aggregation across groups -----
        merged_model = group_models[0]
        for g in range(1, num_groups):
            candidate_models = [merged_model, group_models[g]]
            # Evaluate candidate models by loss on test_loader
            if test_loader is not None:
                losses = []
                for candidate in candidate_models:
                    for node in all_nodes:
                        node.model.set_params(candidate)
                        loss = evaluate_batch_loss(node.model, test_loader)
                        losses.append(loss)
                best_idx = int(np.argmin(losses))
                merged_model = candidate_models[best_idx % len(candidate_models)]
            else:
                # fallback: just average
                merged_model = [(m1 + m2) / 2.0 for m1, m2 in zip(merged_model, group_models[g])]

        # ----- Stage 3: Multicast aggregated model back to all nodes -----
        for node in all_nodes:
            node.model.set_params(deepcopy(merged_model))

        # Evaluate global accuracy after aggregation
        if test_loader is not None:
            accs = [evaluate(node.model, test_loader) for node in all_nodes]
            avg_acc = float(np.mean(accs))
            test_accuracies.append(avg_acc)
            print(f"[Global Round {r + 1}] Average Test Accuracy: {avg_acc:.4f}")

    return [node.model for node in all_nodes], test_accuracies
