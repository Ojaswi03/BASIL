import numpy as np

def gaussian_attack(weights, std=0.01):
    return [w + np.random.normal(0, std, w.shape) for w in weights]

def sign_flip_attack(weights):
    return [-w for w in weights]

def hidden_attack(weights, malicious_weights, blend_ratio=0.5):
    return [(1 - blend_ratio) * w + blend_ratio * mw for w, mw in zip(weights, malicious_weights)]

def apply_attack(weights, attack_type="none", **kwargs):
    if attack_type == "none":
        return weights
    elif attack_type == "gaussian":
        std = kwargs.get("std", 0.01)
        return gaussian_attack(weights, std)
    elif attack_type == "sign_flip":
        return sign_flip_attack(weights)
    elif attack_type == "hidden":
        malicious_weights = kwargs.get("malicious_weights")
        blend_ratio = kwargs.get("blend_ratio", 0.5)
        if malicious_weights is None:
            raise ValueError("Hidden attack requires 'malicious_weights'.")
        return hidden_attack(weights, malicious_weights, blend_ratio)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
