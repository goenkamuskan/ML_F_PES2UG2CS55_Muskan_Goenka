# lab.py
import torch
import numpy as np

def get_entropy_of_dataset(data: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    """
    # Check if the input is a numpy array and convert it to a data
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    labels = data[:, -1]  
    values, counts = torch.unique(labels, return_counts=True)
    probabilities = counts.float() / counts.sum()
    entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-9))  # add eps to avoid log2(0)
    return float(entropy)


def get_avg_info_of_attribute(data: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v.
    """
    # Check if the input is a numpy array and convert it to a data
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    attribute_values = data[:, attribute]
    unique_vals, counts = torch.unique(attribute_values, return_counts=True)
    total_samples = data.shape[0]

    avg_info = 0.0
    for val, count in zip(unique_vals, counts):
        subset = data[attribute_values == val]
        subset_entropy = get_entropy_of_dataset(subset)
        weight = count.item() / total_samples
        avg_info += weight * subset_entropy

    return float(avg_info)


def get_information_gain(data: torch.Tensor, attribute: int):
    """
    Calculate Information Gain for an attribute.
    Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)
    """
    # Check if the input is a numpy array and convert it to a data
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    dataset_entropy = get_entropy_of_dataset(data)
    avg_info = get_avg_info_of_attribute(data, attribute)
    info_gain = dataset_entropy - avg_info
    return round(float(info_gain), 4)


def get_selected_attribute(data: torch.Tensor):
    """
    Select the best attribute based on highest information gain.
    Returns: (dict of gains, best attribute index)
    """
    # Check if the input is a numpy array and convert it to a data
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    n_attributes = data.shape[1] - 1  # exclude labels column
    gains = {}

    for attr in range(n_attributes):
        gains[attr] = get_information_gain(data, attr)

    best_attr = max(gains, key=gains.get)
    return gains, best_attr