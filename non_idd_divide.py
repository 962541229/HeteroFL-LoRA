# coding: UTF-8
from collections import defaultdict

import numpy as np


def split_dataset_to_clients(contents, labels, data_indexs, config):
    num_clients = getattr(config, "num_clients", 3)
    partition_type = getattr(config, "partition_type", "iid")
    seed = getattr(config, "seed", 42)
    np.random.seed(seed)
    assert len(contents) == len(labels) == len(data_indexs), "contents, labels, data_indexs 长度必须一致"
    if partition_type == "iid":
        return iid_split(contents, labels, data_indexs, num_clients)
    if partition_type == "non_iid":
        alpha = getattr(config, "dirichlet_alpha", 0.5)
        return dirichlet_non_iid_split(contents, labels, data_indexs, num_clients, alpha)
    raise ValueError(f"未知的数据划分方式: {partition_type}")


def iid_split(contents, labels, data_indexs, num_clients):
    indices = np.random.permutation(len(contents))
    client_indices = np.array_split(indices, num_clients)
    client_datasets = []
    for idxs in client_indices:
        client_datasets.append(([contents[i] for i in idxs], [labels[i] for i in idxs], [data_indexs[i] for i in idxs]))
    return client_datasets


def dirichlet_non_iid_split(contents, labels, data_indexs, num_clients, alpha):
    client_indices = [[] for _ in range(num_clients)]
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    for _, idxs in label_to_indices.items():
        idxs = np.array(idxs)
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        split_points = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        split_indices = np.split(idxs, split_points)
        for client_id, client_class_indices in enumerate(split_indices):
            client_indices[client_id].extend(client_class_indices.tolist())
    client_datasets = []
    for c in range(num_clients):
        np.random.shuffle(client_indices[c])
        client_datasets.append(([contents[i] for i in client_indices[c]], [labels[i] for i in client_indices[c]], [data_indexs[i] for i in client_indices[c]]))
    return client_datasets
