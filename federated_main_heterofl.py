# coding: UTF-8

import argparse
import logging
import os
import time
import warnings
from typing import Dict, List

import torch
import torch.nn as nn

from config_utils import DEFAULT_DATA_ROOT, DEFAULT_MODEL_ROOT, resolve_model_path, set_global_seed
from util import read_file, init_seeds, batch_iter
from non_idd_divide import split_dataset_to_clients
from federated_util import process_federated_dataset
from optimizer_fc import build_svf_lora_optimizer
from distilbert_model_SVF import DistilBERT_Config, DistilBERT_SVF_LoRA_Model
from ERNIE_model_SVF import ERNIE_Config, ERNIE_SVF_LoRA_Model
from RoBERTa_model_SVF import RoBERTa_Config, RoBERTa_SVF_LoRA_Model
from public_space import PublicSpaceManager

warnings.filterwarnings("ignore")


class Data_Config(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.data_base_path = args.data_root
        self.data_root_path = os.path.join(self.data_base_path, self.dataset)
        self.train_dir = os.path.join(self.data_root_path, "train.tsv")
        self.val_dir = os.path.join(self.data_root_path, "dev.tsv")
        self.test_dir = os.path.join(self.data_root_path, "test.tsv")

        if self.dataset == "SST-2":
            self.class_list = ["0", "1"]
        elif self.dataset == "MNLI":
            self.class_list = ["contradiction", "entailment", "neutral"]
        elif self.dataset == "QNLI":
            self.class_list = ["entailment", "not_entailment"]
        elif self.dataset == "QQP":
            self.class_list = ["0", "1"]
        elif self.dataset == "RTE":
            self.class_list = ["entailment", "not_entailment"]
        else:
            raise ValueError(f"Illegal dataset: {self.dataset}")
        self.cat_to_id = dict(zip(self.class_list, range(len(self.class_list))))
        self.num_classes = len(self.class_list)
        self.num_clients = 3
        self.partition_type = args.partition_type
        self.dirichlet_alpha = args.dirichlet_alpha
        self.seed = args.seed

        self.model_root_path = args.model_root
        self.public_model_name = args.public_model_name
        self.public_bert_path = resolve_model_path(self.model_root_path, self.public_model_name)

        self.svf_r = args.svf_r
        self.svf_alpha = args.svf_alpha
        self.svf_dropout = args.svf_dropout
        self.federated_rounds = args.federated_rounds
        self.refine_public_space = not args.no_refine_public_space
        self.public_bridge_mode = args.public_bridge_mode
        self.local_steps = {0: args.distilbert_steps, 1: args.ernie_steps, 2: args.roberta_steps}
        self.eval_every_round = args.eval_every_round
        self.save_dir = args.save_dir
        self.log_dir = args.log_dir
        self.reset_optimizer_state_after_broadcast = args.reset_optimizer_state_after_broadcast


def parse_args():
    parser = argparse.ArgumentParser(description="Run HeteroFL-LoRA with a BERT public space.")
    parser.add_argument("--dataset", default="SST-2", choices=["SST-2", "MNLI", "QNLI", "QQP", "RTE"])
    parser.add_argument("--data_root", default="/home/students/wzj_4090_2/heterofl_lora_runnable/data")
    parser.add_argument("--model_root", default="/home/students/wzj_4090_2/code_server/..bert-model")
    parser.add_argument("--public_model_name", default="bert-base-uncased")
    parser.add_argument("--distilbert_model_name", default="distilbert-base-uncased")
    parser.add_argument("--ernie_model_name", default="ernie-2-base-en")
    parser.add_argument("--roberta_model_name", default="roberta-large-cased")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=180)
    parser.add_argument("--svf_r", type=int, default=256)
    parser.add_argument("--svf_alpha", type=float, default=32.0)
    parser.add_argument("--svf_dropout", type=float, default=0.1)
    parser.add_argument("--federated_rounds", type=int, default=200)
    parser.add_argument("--distilbert_steps", type=int, default=27)
    parser.add_argument("--ernie_steps", type=int, default=16)
    parser.add_argument("--roberta_steps", type=int, default=5)
    parser.add_argument("--eval_every_round", type=int, default=1)
    parser.add_argument("--partition_type", default="iid", choices=["iid", "non_iid"])
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--public_bridge_mode", default="row_resize_qr", choices=["identity_or_fail", "row_resize", "row_resize_qr"])
    parser.add_argument("--no_refine_public_space", action="store_true")
    parser.add_argument("--reset_optimizer_state_after_broadcast", action="store_true")
    parser.add_argument("--save_dir", default="./saved_dict")
    parser.add_argument("--log_dir", default="./logs_heterofl")
    parser.add_argument("--quick_test", action="store_true", help="Run 2 rounds and 1 local step per client for pipeline debugging.")
    args = parser.parse_args()
    if args.quick_test:
        args.federated_rounds = 2
        args.distilbert_steps = 1
        args.ernie_steps = 1
        args.roberta_steps = 1
        args.eval_every_round = 1
    return args


def build_logger(name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    log_path = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


@torch.no_grad()
def evaluate_one_client(client_id, model, model_config, dev_data, loss_fn, logger=None):
    model.eval()
    x_dev, y_dev, length_dev, mask_dev = dev_data
    batch_dev = batch_iter(x_dev, y_dev, length_dev, mask_dev, batch_size=model_config.batch_size, shuffle=False)
    total_loss = 0.0
    total_correct = 0
    total_num = 0
    for x_batch, y_batch, length_batch, mask_batch in batch_dev:
        x_batch = torch.LongTensor(x_batch).to(model_config.device)
        y_batch = torch.LongTensor(y_batch).to(model_config.device)
        mask_batch = torch.LongTensor(mask_batch).to(model_config.device)
        logits = model(x_batch, mask_batch)
        if isinstance(logits, tuple):
            logits = logits[-1]
        loss = loss_fn(logits, y_batch)
        pred = torch.argmax(logits, dim=1)
        total_loss += loss.item() * y_batch.size(0)
        total_correct += (pred == y_batch).sum().item()
        total_num += y_batch.size(0)
    avg_loss = total_loss / max(1, total_num)
    avg_acc = total_correct / max(1, total_num)
    if logger is not None:
        logger.info(f"Eval | Dev Loss {avg_loss:.4f} | Dev Acc {avg_acc:.4f}")
    return avg_loss, avg_acc


def make_batch_iterator(train_data, model_config):
    x_train, y_train, length_train, mask_train = train_data
    return batch_iter(x_train, y_train, length_train, mask_train, batch_size=model_config.batch_size, shuffle=True)


def next_train_batch(client_runtime):
    try:
        return next(client_runtime["train_iter"])
    except StopIteration:
        client_runtime["local_epoch"] += 1
        client_runtime["train_iter"] = make_batch_iterator(client_runtime["train_data"], client_runtime["model_config"])
        return next(client_runtime["train_iter"])


def train_client_local_steps(client_runtime, loss_fn, local_steps):
    client_id = client_runtime["client_id"]
    model = client_runtime["model"]
    model_config = client_runtime["model_config"]
    optimizer = client_runtime["optimizer"]
    logger = client_runtime["logger"]
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0
    for _ in range(int(local_steps)):
        x_batch, y_batch, length_batch, mask_batch = next_train_batch(client_runtime)
        x_batch = torch.LongTensor(x_batch).to(model_config.device)
        y_batch = torch.LongTensor(y_batch).to(model_config.device)
        mask_batch = torch.LongTensor(mask_batch).to(model_config.device)
        logits = model(x_batch, mask_batch)
        if isinstance(logits, tuple):
            logits = logits[-1]
        loss = loss_fn(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        if hasattr(model_config, "max_grad_norm"):
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=model_config.max_grad_norm)
        optimizer.step()
        client_runtime["global_step"] += 1
        pred = torch.argmax(logits, dim=1)
        total_loss += loss.item() * y_batch.size(0)
        total_correct += (pred == y_batch).sum().item()
        total_num += y_batch.size(0)
    train_loss = total_loss / max(1, total_num)
    train_acc = total_correct / max(1, total_num)
    logger.info(f"Client {client_id} local steps finished | Steps {local_steps} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | Global Step {client_runtime['global_step']}")
    return train_loss, train_acc


def clear_trainable_optimizer_state(optimizer):
    optimizer.state.clear()


def build_clients(args, data_config, processed_client_train_datasets, processed_client_dev_datasets):
    client_settings = [
        {"client_id": 0, "gpu_id": 0, "model_class": DistilBERT_SVF_LoRA_Model, "config_class": DistilBERT_Config, "model_name": args.distilbert_model_name, "train_data": processed_client_train_datasets[0], "dev_data": processed_client_dev_datasets[0]},
        {"client_id": 1, "gpu_id": 1, "model_class": ERNIE_SVF_LoRA_Model, "config_class": ERNIE_Config, "model_name": args.ernie_model_name, "train_data": processed_client_train_datasets[1], "dev_data": processed_client_dev_datasets[1]},
        {"client_id": 2, "gpu_id": 2, "model_class": RoBERTa_SVF_LoRA_Model, "config_class": RoBERTa_Config, "model_name": args.roberta_model_name, "train_data": processed_client_train_datasets[2], "dev_data": processed_client_dev_datasets[2]},
    ]
    clients = []
    os.makedirs(data_config.save_dir, exist_ok=True)
    for setting in client_settings:
        client_id = setting["client_id"]
        logger = build_logger(f"client_{client_id}", data_config.log_dir)
        model_config = setting["config_class"](model_root_path=data_config.model_root_path, model_name=setting["model_name"], gpu_id=setting["gpu_id"], seq_length=args.seq_length)
        model_config.svf_r = data_config.svf_r
        model_config.svf_alpha = data_config.svf_alpha
        model_config.svf_dropout = data_config.svf_dropout
        model_config.save_path = os.path.join(data_config.save_dir, f"heterofl_client_{client_id}_{model_config.model_name.replace('/', '_')}.ckpt")
        print(f"[Client {client_id}] Building {model_config.model_name} on {model_config.device}; path={model_config.bert_path}", flush=True)
        logger.info(f"Building model on {model_config.device}; path={model_config.bert_path}")
        model = setting["model_class"](data_config, model_config).to(model_config.device)
        optimizer, scheduler = build_svf_lora_optimizer(model, model_config)
        clients.append({
            "client_id": client_id,
            "model": model,
            "model_config": model_config,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "train_data": setting["train_data"],
            "dev_data": setting["dev_data"],
            "train_iter": make_batch_iterator(setting["train_data"], model_config),
            "local_epoch": 0,
            "global_step": 0,
            "best_dev_acc": 0.0,
            "logger": logger,
            "num_train_samples": len(setting["train_data"][1]),
        })
    return clients


def run_heterofl_training(args, data_config, processed_client_train_datasets, processed_client_dev_datasets):
    print("Initializing public space from:", data_config.public_bert_path, flush=True)
    server = PublicSpaceManager.from_public_model(
        public_model_path=data_config.public_bert_path,
        rank=data_config.svf_r,
        device="cpu",
        verbose=True,
        bridge_mode=data_config.public_bridge_mode,
    )
    clients = build_clients(args, data_config, processed_client_train_datasets, processed_client_dev_datasets)
    loss_fn = nn.CrossEntropyLoss()

    for client in clients:
        print(f"[Client {client['client_id']}] Matching SVF layers to public spaces...", flush=True)
        client["match_map"] = server.match_client_model(client["model"], verbose=True)
        client["logger"].info(f"Matched {len(client['match_map'])} SVF modules to public spaces.")

    for client in clients:
        server.broadcast_to_client(client["model"], client["match_map"])
        if data_config.reset_optimizer_state_after_broadcast:
            clear_trainable_optimizer_state(client["optimizer"])

    for round_id in range(1, data_config.federated_rounds + 1):
        print(f"\n========== HeteroFL Round {round_id}/{data_config.federated_rounds} ==========", flush=True)
        client_updates = []
        for client in clients:
            client_id = client["client_id"]
            local_steps = int(data_config.local_steps.get(client_id, 1))
            train_loss, train_acc = train_client_local_steps(client, loss_fn, local_steps)
            projected_updates = server.collect_projected_client_updates(
                client_model=client["model"],
                match_map=client["match_map"],
                sample_weight=float(client["num_train_samples"]),
            )
            client_updates.append(projected_updates)
            print(f"[Round {round_id}] Client {client_id} uploaded | Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f}", flush=True)

        update_counts = server.aggregate_and_refine(client_updates=client_updates, refine=data_config.refine_public_space)
        print(f"[Round {round_id}] Server aggregated {sum(update_counts.values())} projected layer updates.", flush=True)

        for client in clients:
            server.broadcast_to_client(client["model"], client["match_map"])
            if data_config.reset_optimizer_state_after_broadcast:
                clear_trainable_optimizer_state(client["optimizer"])

        if round_id % data_config.eval_every_round == 0:
            for client in clients:
                client_id = client["client_id"]
                dev_loss, dev_acc = evaluate_one_client(client_id, client["model"], client["model_config"], client["dev_data"], loss_fn, logger=client["logger"])
                msg = f"[Round {round_id}] Client {client_id} Dev Acc {dev_acc:.4f} | Dev Loss {dev_loss:.4f}"
                print(msg, flush=True)
                client["logger"].info(msg)
                if dev_acc > client["best_dev_acc"]:
                    client["best_dev_acc"] = dev_acc
                    torch.save(client["model"].state_dict(), client["model_config"].save_path)
                    best_msg = f"[Client {client_id}] New Best | Round {round_id} | Dev Acc {dev_acc:.4f} | Saved to {client['model_config'].save_path}"
                    print(best_msg, flush=True)
                    client["logger"].info(best_msg)

        for client in clients:
            client["scheduler"].step()

    print("\n========== HeteroFL-LoRA Training Finished ==========", flush=True)
    for client in clients:
        print(f"Client {client['client_id']} best dev acc: {client['best_dev_acc']:.4f}", flush=True)


def main():
    args = parse_args()
    init_seeds(args.seed)
    set_global_seed(args.seed)
    start_time = time.time()
    data_config = Data_Config(args)
    print("Dataset:", data_config.dataset, flush=True)
    print("Classes:", data_config.class_list, flush=True)
    print("Train file:", data_config.train_dir, flush=True)
    print("Dev file:", data_config.val_dir, flush=True)
    print("Public model:", data_config.public_bert_path, flush=True)

    print("Loading train data...", flush=True)
    train_contents, train_labels, train_data_indexs = read_file(data_config, "train")
    print("Loading dev data...", flush=True)
    dev_contents, dev_labels, dev_data_indexs = read_file(data_config, "dev")
    client_train_datasets = split_dataset_to_clients(train_contents, train_labels, train_data_indexs, data_config)
    client_dev_datasets = [(dev_contents, dev_labels, dev_data_indexs) for _ in range(3)]

    distilbert_config = DistilBERT_Config(model_root_path=args.model_root, model_name=args.distilbert_model_name, gpu_id=0, seq_length=args.seq_length)
    ernie_config = ERNIE_Config(model_root_path=args.model_root, model_name=args.ernie_model_name, gpu_id=1, seq_length=args.seq_length)
    roberta_config = RoBERTa_Config(model_root_path=args.model_root, model_name=args.roberta_model_name, gpu_id=2, seq_length=args.seq_length)
    for cfg in [distilbert_config, ernie_config, roberta_config]:
        cfg.svf_r = data_config.svf_r
        cfg.svf_alpha = data_config.svf_alpha
        cfg.svf_dropout = data_config.svf_dropout
    client_model_configs = [distilbert_config, ernie_config, roberta_config]

    print("Processing federated train data...", flush=True)
    processed_client_train_datasets = process_federated_dataset(data_config, client_train_datasets, client_model_configs, parallel=True, show_progress=False)
    print("Processing federated dev data...", flush=True)
    processed_client_dev_datasets = process_federated_dataset(data_config, client_dev_datasets, client_model_configs, parallel=True, show_progress=False)

    run_heterofl_training(args, data_config, processed_client_train_datasets, processed_client_dev_datasets)
    print("Total time used:", time.time() - start_time, flush=True)


if __name__ == "__main__":
    main()
