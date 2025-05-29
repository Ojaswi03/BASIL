import numpy as np
import random
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Subset

def load_mnist(iid, num_users, seed):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = MNIST(root='./data', train=False, download=True, transform=transform)

    if iid:
        user_data = _iid_partition(train_set, num_users, seed)
    else:
        user_data = _non_iid_partition(train_set, num_users, seed)

    return user_data, test_set

def _iid_partition(dataset, num_users, seed):
    num_items = int(len(dataset) / num_users)
    all_idxs = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(all_idxs)
    return {i: Subset(dataset, all_idxs[i * num_items:(i + 1) * num_items]) for i in range(num_users)}

def _non_iid_partition(dataset, num_users, seed):
    labels = np.array(dataset.targets)
    idxs = np.arange(len(dataset))
    sorted_idxs = idxs[np.argsort(labels)]

    num_shards = 2 * num_users
    num_imgs = int(len(dataset) / num_shards)
    shards = [sorted_idxs[i * num_imgs:(i + 1) * num_imgs] for i in range(num_shards)]

    random.seed(seed)
    random.shuffle(shards)
    user_data = {i: Subset(dataset, np.concatenate(shards[2 * i:2 * i + 2])) for i in range(num_users)}
    return user_data
