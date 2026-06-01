# coding: UTF-8
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from config_utils import DEFAULT_MODEL_ROOT, pick_device, resolve_model_path
from SVFLinear import SVFLinear


class DistilBERT_Config(object):
    def __init__(self, model_root_path=DEFAULT_MODEL_ROOT, model_name="distilbert-base-uncased", gpu_id=0, seq_length=180):
        self.model_root_path = model_root_path
        self.model_name = model_name
        self.save_path = "./saved_dict/" + self.model_name.replace("/", "_") + ".ckpt"
        self.gpu_id = int(gpu_id)
        self.device = pick_device(self.gpu_id)
        self.num_epochs = 60000
        self.batch_size = 64
        self.seq_length = int(seq_length)
        self.print_per_batch = 1
        self.bert_path = resolve_model_path(self.model_root_path, self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path, use_fast=True)
        self.hidden_size = 768
        self.svf_r = 256
        self.svf_alpha = 32
        self.svf_dropout = 0.1
        self.svf_learning_rate = 1e-2
        self.fc_learning_rate = 5e-4
        self.svf_weight_decay = 0.0
        self.fc_weight_decay = 0.01
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-8
        self.scheduler_milestones = [10, 30]
        self.scheduler_gamma = 0.5
        self.max_grad_norm = 1.0


class DistilBERT_SVF_LoRA_Model(nn.Module):
    def __init__(self, data_config, model_config):
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.bert = AutoModel.from_pretrained(self.model_config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.svf_r = getattr(self.model_config, "svf_r", 256)
        self.svf_alpha = getattr(self.model_config, "svf_alpha", 32)
        self.svf_dropout = getattr(self.model_config, "svf_dropout", 0.1)
        self.svf_target_modules = getattr(self.model_config, "svf_target_modules", ("q_lin", "v_lin", "lin1", "lin2"))
        self.num_svf_layers = 0
        self.inject_svf_lora(self.bert)
        hidden_size = getattr(self.bert.config, "hidden_size", getattr(self.bert.config, "dim", self.model_config.hidden_size))
        self.fc = nn.Linear(hidden_size, self.data_config.num_classes)
        self.print_trainable_parameters()

    def inject_svf_lora(self, module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in self.svf_target_modules:
                setattr(module, name, SVFLinear(child, r=self.svf_r, alpha=self.svf_alpha, dropout=self.svf_dropout))
                self.num_svf_layers += 1
            else:
                self.inject_svf_lora(child)

    def forward(self, x, mask):
        output = self.bert(input_ids=x, attention_mask=mask, return_dict=True)
        text_representation = output.last_hidden_state[:, 0, :]
        out = self.fc(text_representation)
        if getattr(self.model_config, "return_text_representation", False):
            return text_representation, out
        return out

    def iter_svf_modules(self):
        for name, module in self.named_modules():
            if isinstance(module, SVFLinear):
                yield name, module

    def get_svf_parameters(self):
        return {name + ".delta_sigma": m.delta_sigma.detach().cpu() for name, m in self.iter_svf_modules()}

    def get_svf_state(self):
        return {name: m.get_svf_state(detach=True, cpu=True) for name, m in self.iter_svf_modules()}

    @torch.no_grad()
    def set_svf_delta(self, delta_dict, strict=False):
        module_dict = dict(self.iter_svf_modules())
        for name, delta_sigma in delta_dict.items():
            if name not in module_dict:
                if strict:
                    raise KeyError(f"Unknown SVF module: {name}")
                continue
            module_dict[name].set_delta_sigma(delta_sigma)

    @torch.no_grad()
    def reset_svf_delta(self):
        for _, module in self.iter_svf_modules():
            module.reset_delta_sigma()

    def print_trainable_parameters(self):
        trainable_params = 0
        total_params = 0
        for _, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        ratio = 100 * trainable_params / max(1, total_params)
        print(f"Trainable params: {trainable_params} | Total params: {total_params} | Trainable ratio: {ratio:.6f}%")
