# coding: UTF-8


import csv
import os
import random
import time
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import torch


def init_seeds(seed=1):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_time_dif(start_time):
    return timedelta(seconds=int(round(time.time() - start_time)))


def _has_header(first_row):
    joined = "\t".join([str(x).lower() for x in first_row])
    header_keywords = ["sentence", "label", "question", "premise", "hypothesis", "gold_label", "is_duplicate"]
    return any(k in joined for k in header_keywords)


def _find_col(header, candidates):
    lower = [h.lower() for h in header]
    for c in candidates:
        if c.lower() in lower:
            return lower.index(c.lower())
    return None


def _get(row, idx, default=""):
    if idx is None or idx < 0 or idx >= len(row):
        return default
    return row[idx]


def _read_tsv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = [r for r in reader if len(r) > 0]
    if not rows:
        raise RuntimeError(f"Empty data file: {path}")
    return rows


def read_file(config, split="train"):

    split = str(split).lower()
    if split == "train":
        path = config.train_dir
    elif split in {"dev", "val", "valid", "validation"}:
        path = getattr(config, "val_dir", None) or getattr(config, "dev_dir", None)
    elif split == "test":
        path = config.test_dir
    else:
        raise ValueError(f"Unsupported split: {split}")

    rows = _read_tsv(path)
    dataset = str(config.dataset).upper()
    has_header = _has_header(rows[0])
    header = rows[0] if has_header else None
    data_rows = rows[1:] if has_header else rows

    contents, labels, data_indexs = [], [], []

    if has_header:
        # Common GLUE column names.
        label_idx = _find_col(header, ["label", "gold_label", "is_duplicate"])
        if dataset == "SST-2":
            a_idx = _find_col(header, ["sentence", "text", "sentence1"])
            b_idx = None
        elif dataset == "MNLI":
            a_idx = _find_col(header, ["sentence1", "premise"])
            b_idx = _find_col(header, ["sentence2", "hypothesis"])
        elif dataset == "QNLI":
            a_idx = _find_col(header, ["question", "sentence1"])
            b_idx = _find_col(header, ["sentence", "sentence2"])
        elif dataset == "QQP":
            a_idx = _find_col(header, ["question1", "sentence1"])
            b_idx = _find_col(header, ["question2", "sentence2"])
        elif dataset == "RTE":
            a_idx = _find_col(header, ["sentence1", "premise"])
            b_idx = _find_col(header, ["sentence2", "hypothesis"])
        else:
            a_idx = _find_col(header, ["sentence", "text", "sentence1", "question1", "premise"])
            b_idx = _find_col(header, ["sentence2", "question2", "hypothesis"])

        if label_idx is None:
            # For unlabeled test files, use the first class as placeholder.
            label_placeholder = config.class_list[0]
        else:
            label_placeholder = None

        for i, row in enumerate(data_rows):
            label = label_placeholder if label_placeholder is not None else _get(row, label_idx)
            if label == "" or label == "-":
                continue
            text_a = _get(row, a_idx)
            if b_idx is None:
                content = text_a
            else:
                content = (text_a, _get(row, b_idx))
            contents.append(content)
            labels.append(label)
            data_indexs.append(i)
    else:

        for i, row in enumerate(data_rows):
            if len(row) < 2:
                continue
            label = row[-1]
            if dataset == "SST-2":
                content = row[0]
            else:
                text_cols = row[:-1]
                if len(text_cols) >= 3 and text_cols[0].isdigit():
                    text_cols = text_cols[1:]
                if len(text_cols) < 2:
                    continue
                content = (text_cols[-2], text_cols[-1])
            contents.append(content)
            labels.append(label)
            data_indexs.append(i)

    if len(contents) == 0:
        raise RuntimeError(f"No samples parsed from {path}. Check your TSV format.")
    return contents, labels, data_indexs


def process_dataset(*args, **kwargs):
    raise NotImplementedError("Use federated_util.process_federated_dataset for this HeteroFL-LoRA package.")


def batch_iter(x, y, lengths, masks, batch_size=32, shuffle=True):
    x = np.asarray(x)
    y = np.asarray(y)
    lengths = np.asarray(lengths)
    masks = np.asarray(masks)
    n = len(y)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, int(batch_size)):
        batch_idx = indices[start:start + int(batch_size)]
        yield x[batch_idx], y[batch_idx], lengths[batch_idx], masks[batch_idx]
