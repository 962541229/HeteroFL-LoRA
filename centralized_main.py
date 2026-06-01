# coding: UTF-8
"""
用于独立微调每个客户端模型，使用完整训练集，不进行联邦聚合。
"""

import os
import time
import argparse
import logging
import warnings

import torch
import torch.nn as nn

from util import read_file, init_seeds, batch_iter
from federated_util import process_client_dataset
from optimizer_fc import build_svf_lora_optimizer

from distilbert_model_SVF import DistilBERT_Config, DistilBERT_SVF_LoRA_Model
from ERNIE_model_SVF import ERNIE_Config, ERNIE_SVF_LoRA_Model
from RoBERTa_model_SVF import RoBERTa_Config, RoBERTa_SVF_LoRA_Model


warnings.filterwarnings("ignore")


class CentralizedDataConfig(object):
    def __init__(self, dataset="SST-2", data_root=None):
        self.dataset = dataset

        if data_root is None:
            data_root = "/home/students/wzj_4090_2/code_server/A_paper4/data"

        self.data_root = data_root
        self.data_root_path = os.path.join(self.data_root, self.dataset)

        self.train_dir = os.path.join(self.data_root_path, "train.tsv")
        self.val_dir = os.path.join(self.data_root_path, "dev.tsv")
        self.dev_dir = os.path.join(self.data_root_path, "dev.tsv")
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
            raise ValueError(f"非法数据集: {self.dataset}")

        self.cat_to_id = dict(zip(self.class_list, range(len(self.class_list))))
        self.num_classes = len(self.class_list)

        # 为了兼容原先 federated 代码中的 config 访问
        self.num_clients = 3
        self.partition_type = "centralized"
        self.dirichlet_alpha = None


def build_logger(name, log_dir="./logs_centralized"):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    log_path = os.path.join(log_dir, f"{name}.log")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.propagate = False

    return logger


def maybe_limit_dataset(contents, labels, data_indexs, max_samples):

    if max_samples is None or max_samples <= 0:
        return contents, labels, data_indexs

    max_samples = min(max_samples, len(labels))

    return (
        contents[:max_samples],
        labels[:max_samples],
        data_indexs[:max_samples]
    )


def set_runtime_config(model_config, args, model_tag):

    if args.num_epochs is not None:
        model_config.num_epochs = args.num_epochs

    if args.batch_size is not None:
        model_config.batch_size = args.batch_size

    if args.seq_length is not None:
        model_config.seq_length = args.seq_length

    if args.svf_lr is not None:
        model_config.svf_learning_rate = args.svf_lr

    if args.fc_lr is not None:
        model_config.fc_learning_rate = args.fc_lr

    if args.max_grad_norm is not None:
        model_config.max_grad_norm = args.max_grad_norm

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    safe_model_name = model_config.model_name.replace("/", "_")
    model_config.save_path = os.path.join(
        save_dir,
        f"centralized_{args.dataset}_{model_tag}_{safe_model_name}.ckpt"
    )

    return model_config


@torch.no_grad()
def evaluate(model, model_config, dev_data, loss_fn):
    model.eval()

    x_dev, y_dev, length_dev, mask_dev = dev_data

    batch_dev = batch_iter(
        x_dev,
        y_dev,
        length_dev,
        mask_dev,
        batch_size=model_config.batch_size,
        shuffle=False
    )

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
        correct = (pred == y_batch).sum().item()

        total_loss += loss.item() * y_batch.size(0)
        total_correct += correct
        total_num += y_batch.size(0)

    avg_loss = total_loss / max(total_num, 1)
    avg_acc = total_correct / max(total_num, 1)

    return avg_loss, avg_acc


def train_one_epoch(
    model_tag,
    model,
    model_config,
    train_data,
    dev_data,
    optimizer,
    loss_fn,
    epoch,
    global_step,
    best_dev_acc,
    logger,
    args
):
    model.train()

    x_train, y_train, length_train, mask_train = train_data

    batch_train = batch_iter(
        x_train,
        y_train,
        length_train,
        mask_train,
        batch_size=model_config.batch_size,
        shuffle=True
    )

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for batch_id, (x_batch, y_batch, length_batch, mask_batch) in enumerate(batch_train):
        x_batch = torch.LongTensor(x_batch).to(model_config.device)
        y_batch = torch.LongTensor(y_batch).to(model_config.device)
        mask_batch = torch.LongTensor(mask_batch).to(model_config.device)

        logits = model(x_batch, mask_batch)

        if isinstance(logits, tuple):
            logits = logits[-1]

        loss = loss_fn(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()

        if hasattr(model_config, "max_grad_norm") and model_config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=model_config.max_grad_norm
            )

        optimizer.step()
        global_step += 1

        pred = torch.argmax(logits, dim=1)
        correct = (pred == y_batch).sum().item()

        total_loss += loss.item() * y_batch.size(0)
        total_correct += correct
        total_num += y_batch.size(0)

        train_loss = total_loss / max(total_num, 1)
        train_acc = total_correct / max(total_num, 1)

        should_eval = (
            args.eval_every > 0
            and global_step % args.eval_every == 0
        )

        if should_eval:
            dev_loss, dev_acc = evaluate(
                model=model,
                model_config=model_config,
                dev_data=dev_data,
                loss_fn=loss_fn
            )

            log_msg = (
                f"[{model_tag}] "
                f"Epoch {epoch} | Step {global_step} | Batch {batch_id} | "
                f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
                f"Dev Loss {dev_loss:.4f} | Dev Acc {dev_acc:.4f}"
            )
            logger.info(log_msg)

            if args.print_every > 0 and global_step % args.print_every == 0:
                print(log_msg, flush=True)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                torch.save(model.state_dict(), model_config.save_path)

                best_msg = (
                    f"[{model_tag}] New Best! "
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Dev Acc {dev_acc:.4f} | Dev Loss {dev_loss:.4f} | "
                    f"Saved to {model_config.save_path}"
                )

                print(best_msg, flush=True)
                logger.info(best_msg)

            model.train()

    epoch_loss = total_loss / max(total_num, 1)
    epoch_acc = total_correct / max(total_num, 1)

    return epoch_loss, epoch_acc, global_step, best_dev_acc


def train_one_model(
    model_tag,
    model_class,
    model_config_class,
    gpu_id,
    data_config,
    raw_train_data,
    raw_dev_data,
    args
):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    logger = build_logger(f"centralized_{model_tag}", args.log_dir)

    model_config = model_config_class()
    model_config.gpu_id = gpu_id
    model_config.device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    model_config = set_runtime_config(
        model_config=model_config,
        args=args,
        model_tag=model_tag
    )

    logger.info("=" * 80)
    logger.info(f"Start centralized training for {model_tag}")
    logger.info(f"Model name: {model_config.model_name}")
    logger.info(f"Device: {model_config.device}")
    logger.info(f"Epochs: {model_config.num_epochs}")
    logger.info(f"Batch size: {model_config.batch_size}")
    logger.info(f"Seq length: {model_config.seq_length}")
    logger.info(f"Save path: {model_config.save_path}")

    print("=" * 80, flush=True)
    print(f"[{model_tag}] Start centralized full-data fine-tuning", flush=True)
    print(f"[{model_tag}] Model: {model_config.model_name}", flush=True)
    print(f"[{model_tag}] Device: {model_config.device}", flush=True)

    train_contents, train_labels, train_indexs = raw_train_data
    dev_contents, dev_labels, dev_indexs = raw_dev_data

    print(f"[{model_tag}] Tokenizing train data with {model_config.model_name} ...", flush=True)
    train_data = process_client_dataset(
        data_config=data_config,
        model_config=model_config,
        contents=train_contents,
        labels=train_labels,
        data_indexs=train_indexs,
        show_progress=args.show_progress
    )

    print(f"[{model_tag}] Tokenizing dev data with {model_config.model_name} ...", flush=True)
    dev_data = process_client_dataset(
        data_config=data_config,
        model_config=model_config,
        contents=dev_contents,
        labels=dev_labels,
        data_indexs=dev_indexs,
        show_progress=args.show_progress
    )

    print(f"[{model_tag}] Building model ...", flush=True)
    model = model_class(data_config, model_config)
    model = model.to(model_config.device)

    optimizer, scheduler = build_svf_lora_optimizer(model, model_config)
    loss_fn = nn.CrossEntropyLoss()

    global_step = 0
    best_dev_acc = 0.0

    start_time = time.time()

    for epoch in range(model_config.num_epochs):
        epoch_loss, epoch_acc, global_step, best_dev_acc = train_one_epoch(
            model_tag=model_tag,
            model=model,
            model_config=model_config,
            train_data=train_data,
            dev_data=dev_data,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch=epoch,
            global_step=global_step,
            best_dev_acc=best_dev_acc,
            logger=logger,
            args=args
        )

        scheduler.step()

        epoch_msg = (
            f"[{model_tag}] Epoch {epoch} finished | "
            f"Train Loss {epoch_loss:.4f} | Train Acc {epoch_acc:.4f} | "
            f"Best Dev Acc {best_dev_acc:.4f}"
        )

        print(epoch_msg, flush=True)
        logger.info(epoch_msg)

    total_time = time.time() - start_time

    final_msg = (
        f"[{model_tag}] Training finished | "
        f"Best Dev Acc {best_dev_acc:.4f} | "
        f"Time {total_time:.2f}s | "
        f"Best model: {model_config.save_path}"
    )

    print(final_msg, flush=True)
    logger.info(final_msg)

    return {
        "model_tag": model_tag,
        "model_name": model_config.model_name,
        "best_dev_acc": best_dev_acc,
        "save_path": model_config.save_path,
        "time": total_time
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="SST-2",
        choices=["SST-2", "MNLI", "QNLI", "QQP", "RTE"]
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/students/wzj_4090_2/code_server/A_paper4/data"
    )

    parser.add_argument(
        "--clients",
        type=str,
        default="roberta",
        help="可选：distilbert,ernie,roberta，例如 --clients distilbert 或 --clients distilbert,roberta"
    )

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seq_length", type=int, default=None)

    parser.add_argument("--svf_lr", type=float, default=None)
    parser.add_argument("--fc_lr", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)

    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="每多少个 optimizer.step 在 dev 上验证一次；设为 0 表示不做 step-level 验证"
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="每多少个 step 打印一次普通训练日志；best 结果始终打印"
    )

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved_dict"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs_centralized"
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="快速测试用；0 表示使用完整训练集"
    )

    parser.add_argument(
        "--max_dev_samples",
        type=int,
        default=0,
        help="快速测试用；0 表示使用完整验证集"
    )

    parser.add_argument(
        "--show_progress",
        action="store_true"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    init_seeds(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    data_config = CentralizedDataConfig(
        dataset=args.dataset,
        data_root=args.data_root
    )

    print("=" * 80, flush=True)
    print("Centralized full-data SVF-LoRA fine-tuning", flush=True)
    print(f"Dataset: {data_config.dataset}", flush=True)
    print(f"Train path: {data_config.train_dir}", flush=True)
    print(f"Dev path: {data_config.dev_dir}", flush=True)
    print(f"Classes: {data_config.class_list}", flush=True)
    print("=" * 80, flush=True)

    print("Loading full train data ...", flush=True)
    train_contents, train_labels, train_indexs = read_file(data_config, "train")

    print("Loading full dev data ...", flush=True)
    dev_contents, dev_labels, dev_indexs = read_file(data_config, "dev")

    train_contents, train_labels, train_indexs = maybe_limit_dataset(
        train_contents,
        train_labels,
        train_indexs,
        args.max_train_samples
    )

    dev_contents, dev_labels, dev_indexs = maybe_limit_dataset(
        dev_contents,
        dev_labels,
        dev_indexs,
        args.max_dev_samples
    )

    print(f"Train samples: {len(train_labels)}", flush=True)
    print(f"Dev samples: {len(dev_labels)}", flush=True)

    raw_train_data = (train_contents, train_labels, train_indexs)
    raw_dev_data = (dev_contents, dev_labels, dev_indexs)

    requested_clients = [
        item.strip().lower()
        for item in args.clients.split(",")
        if item.strip()
    ]

    client_registry = {
        "distilbert": {
            "model_tag": "distilbert",
            "model_class": DistilBERT_SVF_LoRA_Model,
            "model_config_class": DistilBERT_Config,
            "gpu_id": 0
        },
        "ernie": {
            "model_tag": "ernie",
            "model_class": ERNIE_SVF_LoRA_Model,
            "model_config_class": ERNIE_Config,
            "gpu_id": 1
        },
        "roberta": {
            "model_tag": "roberta",
            "model_class": RoBERTa_SVF_LoRA_Model,
            "model_config_class": RoBERTa_Config,
            "gpu_id": 2
        }
    }

    results = []

    for client_name in requested_clients:
        if client_name not in client_registry:
            raise ValueError(
                f"未知 client: {client_name}. "
                f"可选值: {list(client_registry.keys())}"
            )

        setting = client_registry[client_name]

        result = train_one_model(
            model_tag=setting["model_tag"],
            model_class=setting["model_class"],
            model_config_class=setting["model_config_class"],
            gpu_id=setting["gpu_id"],
            data_config=data_config,
            raw_train_data=raw_train_data,
            raw_dev_data=raw_dev_data,
            args=args
        )

        results.append(result)

    print("=" * 80, flush=True)
    print("Centralized training summary", flush=True)

    for result in results:
        print(
            f"{result['model_tag']} | "
            f"{result['model_name']} | "
            f"Best Dev Acc: {result['best_dev_acc']:.4f} | "
            f"Save: {result['save_path']} | "
            f"Time: {result['time']:.2f}s",
            flush=True
        )

    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()