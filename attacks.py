import numpy as np
import random


def gaussian_attack(weights, mean=0.0, std=1.0):
    """
    Gaussian attack: replace weights with Gaussian noise.
    """
    return [np.random.normal(mean, std, w.shape).astype(np.float32) for w in weights]


def sign_flip_attack(weights):
    """
    Random Sign-Flip Attack (layer-wise):
    For each layer, randomly decide whether to flip the entire layer.
    """
    attacked = []
    for w in weights:
        if random.random() < 0.5:
            attacked.append((-1.0 * w).astype(np.float32))
        else:
            attacked.append(w.astype(np.float32))
    return attacked


def hidden_attack(weights, malicious_weights=None, blend_ratio=0.5):
    """
    Hidden attack: craft malicious weights close to benign ones.
    Blend benign weights with adversarial (malicious) weights.
    """
    if malicious_weights is None:
        # fallback: just return original weights (no attack possible)
        return weights
    attacked = []
    for w, m in zip(weights, malicious_weights):
        blend = (1 - blend_ratio) * w + blend_ratio * m
        attacked.append(blend.astype(np.float32))
    return attacked


def apply_attack(weights, attack_type, malicious_weights=None, blend_ratio=0.5):
    """
    Dispatch function to apply a specified attack type.
    """
    if attack_type == 'gaussian':
        return gaussian_attack(weights)
    elif attack_type == 'sign_flip':
        return sign_flip_attack(weights)
    elif attack_type == 'hidden':
        return hidden_attack(weights, malicious_weights, blend_ratio)
    elif attack_type == 'none':
        return weights
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
