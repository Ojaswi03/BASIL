import numpy as np
import random


def split_sensitive_non_sensitive(dataset, alpha=0.05, seed=42):
    """
    Split dataset indices into sensitive and non-sensitive.
    """
    random.seed(seed)
    total_len = len(dataset)
    indices = list(range(total_len))
    random.shuffle(indices)
    split_point = int(alpha * total_len)
    return indices[split_point:], indices[:split_point]


def partition_batches(non_sensitive_indices, H):
    """
    Partition non-sensitive data indices into H batches.
    """
    random.shuffle(non_sensitive_indices)
    batch_size = max(1, len(non_sensitive_indices) // H)
    batches = []
    for i in range(H):
        start = i * batch_size
        end = (i + 1) * batch_size if i < H - 1 else len(non_sensitive_indices)
        batches.append(non_sensitive_indices[start:end])
    return batches


def acds_share(group_nodes, group_data, alpha=0.05, H=5, dummy_batch=None, G=1):
    """
    Full implementation of ACDS:
      Phase 1: Initialization (split sensitive/non-sensitive, partition batches)
      Phase 2: Within-group anonymous cyclic sharing
      Phase 3: Global sharing (merge across groups)

    Args:
        group_nodes: list of node IDs
        group_data: dict {node_id: dataset indices}
        alpha: fraction of data to share
        H: number of batches per node
        dummy_batch: optional dummy indices (e.g., from public dataset)
        G: number of groups (default 1)

    Returns:
        dict {node_id: list of shared indices}
    """
    # Phase 1: Initialization
    node_batches = {}
    for node in group_nodes:
        _, non_sensitive = split_sensitive_non_sensitive(group_data[node], alpha)
        node_batches[node] = partition_batches(non_sensitive, H)

    # Each node stores what it has collected
    node_storage = {node: [] for node in group_nodes}

    # Phase 2: Within-group cyclic sharing
    for h in range(H):
        circulating = []
        for node in group_nodes:
            # add this node's h-th batch to the circulating pool
            circulating.extend(node_batches[node][h])
            # shuffle before sending forward
            random.shuffle(circulating)
            # store what has been seen so far
            node_storage[node].extend(circulating)

    # Dummy round to ensure fairness
    if dummy_batch is not None:
        for node in group_nodes:
            node_storage[node].extend(dummy_batch)

    # Phase 3: Global sharing (simulate node 1 broadcasting to all)
    if G > 1:
        global_data = []
        for node in group_nodes:
            global_data.extend(node_storage[node])
        for node in group_nodes:
            node_storage[node].extend(global_data)

    return node_storage


def apply_acds(dataset, shared_indices):
    """
    Apply ACDS: return injected data samples corresponding to shared indices.
    """
    injected = [dataset[idx] for idx in shared_indices if idx < len(dataset)]
    return injected
