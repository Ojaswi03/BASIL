import numpy as np

def gaussian_attack(weights, std):
    print(f"Applying Gaussian attack with std={std}")
    return [w + np.random.normal(0, std, w.shape) for w in weights]

def sign_flip_attack(weights):
    return [-w for w in weights]

def hidden_attack(weights, malicious_weights, blend_ratio):
    return [(1 - blend_ratio) * w + blend_ratio * mw for w, mw in zip(weights, malicious_weights)]

# def apply_attack(weights, attack_type, std=1.0, malicious_weights=None, blend_ratio=0.5):
#     if attack_type == "none":
#         return weights
#     if attack_type == "gaussian":
#         return gaussian_attack(weights, std)
#     if attack_type == "sign_flip":
#         return sign_flip_attack(weights)
#     if attack_type == "hidden":
#         if malicious_weights is None:
#             raise ValueError("Hidden attack requires 'malicious_weights'.")
#         return hidden_attack(weights, malicious_weights, blend_ratio)
#     raise ValueError(f"Unknown attack type: {attack_type}")



def apply_attack(weights, attack_type):
    attacked_weights = []

    for w in weights:
        flat = w.flatten()

        if attack_type == 'gaussian':
            noise = np.random.normal(loc=0.0, scale=1.0, size=flat.shape)
            print("🔧 Applying Gaussian Noise Attack (std=1.0)")
            attacked_w = (flat + noise).reshape(w.shape)
            

        elif attack_type == 'sign_flip':
            print("🔧 Applying Sign Flip Attack")
            attacked_w = (-flat).reshape(w.shape)

        elif attack_type == 'hidden':
            print("🔧 Applying Hidden Attack")
            zero_out = np.zeros_like(flat)
            zero_out[:len(flat)//3] = flat[:len(flat)//3]
            attacked_w = zero_out.reshape(w.shape)

        else:
            attacked_w = w

        attacked_weights.append(attacked_w)

    return attacked_weights

