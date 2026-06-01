# coding: UTF-8
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm


def process_client_dataset(data_config, model_config, contents, labels, data_indexs, show_progress=False):
    data_id, text_lengths, label_id, masks = [], [], [], []
    iterator = range(len(labels))
    if show_progress:
        iterator = tqdm(iterator, desc=f"Tokenizing {model_config.model_name}")

    for i in iterator:
        content = contents[i]
        if isinstance(content, tuple):
            text_a, text_b = content
            token = model_config.tokenizer.encode_plus(text_a, text_b, truncation=True, padding="max_length", max_length=model_config.seq_length)
        else:
            token = model_config.tokenizer.encode_plus(content, truncation=True, padding="max_length", max_length=model_config.seq_length)

        token_ids = token["input_ids"]
        attention_mask = token["attention_mask"]
        seq_len = attention_mask.count(1)
        data_id.append(token_ids)
        text_lengths.append(seq_len)
        masks.append(attention_mask)
        label = labels[i]
        if isinstance(label, (int, np.integer)):
            label_id.append(int(label))
        else:
            label_id.append(data_config.cat_to_id[str(label)])

    return np.array(data_id), np.array(label_id), np.array(text_lengths), np.array(masks)


def process_federated_dataset(data_config, client_datasets, client_model_configs, parallel=True, show_progress=False):
    assert len(client_datasets) == len(client_model_configs), "客户端数据数量必须和 model_config 数量一致"

    def process_one_client(client_id):
        contents, labels, data_indexs = client_datasets[client_id]
        model_config = client_model_configs[client_id]
        print(f"Processing Client {client_id} with tokenizer: {model_config.model_name}", flush=True)
        result = process_client_dataset(data_config, model_config, contents, labels, data_indexs, show_progress=False if parallel else show_progress)
        print(f"Client {client_id} tokenizing finished. Samples: {len(labels)}", flush=True)
        return result

    if parallel:
        with ThreadPoolExecutor(max_workers=len(client_datasets)) as executor:
            return list(executor.map(process_one_client, range(len(client_datasets))))
    return [process_one_client(client_id) for client_id in range(len(client_datasets))]
