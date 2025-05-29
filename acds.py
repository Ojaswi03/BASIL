import numpy as np
import random

def split_sensitive_non_sensitive(dataset, alpha=0.05, seed=42):
    random.seed(seed)
    total_len = len(dataset)
    indices = list(range(total_len))
    random.shuffle(indices)
    split_point = int(alpha * total_len)
    return indices[split_point:], indices[:split_point]

def partition_batches(non_sensitive_indices, H):
    random.shuffle(non_sensitive_indices)
    batch_size = len(non_sensitive_indices) // H
    batches = []
    for i in range(H):
        start = i * batch_size
        end = (i + 1) * batch_size if i < H - 1 else len(non_sensitive_indices)
        batches.append(non_sensitive_indices[start:end])
    return batches

def acds_share(group_nodes, group_data, alpha=0.05, H=5, dummy_batch=[]):
    group_shared = {node: [] for node in group_nodes}
    node_batches = {}

    for node in group_nodes:
        _, non_sensitive = split_sensitive_non_sensitive(group_data[node], alpha)
        node_batches[node] = partition_batches(non_sensitive, H)

    for h in range(H):
        shared_batches = []
        for node in group_nodes:
            shared_batches.append((node, node_batches[node][h]))

        random.shuffle(shared_batches)
        combined = []
        for _, batch in shared_batches:
            combined.extend(batch)

        for node in group_nodes:
            group_shared[node].extend(combined)

    # Dummy round: anonymize missing last round contributions
    if dummy_batch:
        for node in group_nodes:
            group_shared[node].extend(dummy_batch)

    return group_shared

def apply_acds(dataset, shared_indices):
    injected = [dataset[idx] for idx in shared_indices if idx < len(dataset)]
    return injected
