import numpy as np
import random

def split_sensitive_non_sensitive(dataset, alpha=0.05, seed=42):
    """
    Split dataset into sensitive and non-sensitive parts.
    alpha: fraction of data to be considered non-sensitive and shared.
    """
    random.seed(seed)
    total_len = len(dataset)
    indices = list(range(total_len))
    random.shuffle(indices)
    split_point = int(alpha * total_len)
    return indices[split_point:], indices[:split_point]  # sensitive, non-sensitive

def partition_batches(non_sensitive_indices, H):
    """
    Partition non-sensitive indices into H equal-sized batches.
    """
    random.shuffle(non_sensitive_indices)
    batch_size = len(non_sensitive_indices) // H
    return [non_sensitive_indices[i * batch_size:(i + 1) * batch_size] for i in range(H)]

def acds_share(group_nodes, group_data, alpha=0.05, H=5, seed=42):
    """
    Anonymous Cyclic Data Sharing for a group of nodes
    group_nodes: list of node IDs in this group
    group_data: dict mapping node_id -> dataset (torchvision-style)
    """
    group_shared = {i: [] for i in group_nodes}
    node_batches = {}

    for node in group_nodes:
        _, non_sensitive = split_sensitive_non_sensitive(group_data[node], alpha, seed)
        node_batches[node] = partition_batches(non_sensitive, H)

    for h in range(H):
        shared_pool = []
        for node in group_nodes:
            shared_pool.extend(node_batches[node][h])
        random.shuffle(shared_pool)
        for node in group_nodes:
            group_shared[node].extend(shared_pool)

    return group_shared  # node_id -> shared indices

def apply_acds(data, shared_indices):
    """
    data: full dataset
    shared_indices: list of indices to inject into local dataset
    """
    injected = [data[idx] for idx in shared_indices]
    return injected
