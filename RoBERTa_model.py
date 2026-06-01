import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class RoBERTa_Config(object):
    """RoBERTa 模型配置参数"""
    def __init__(self):
        self.model_root_path = "/home/students/wzj_4090_2/code_server/..bert-model/"
        self.model_name = "roberta-base-cased"

        self.save_path = './saved_dict/' + self.model_name + '.ckpt'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_epochs = 60000
        self.batch_size = 64
        self.seq_length = 180
        self.print_per_batch = 1

        self.bert_path = self.model_root_path + self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)

        # roberta-base 是 768，roberta-large 是 1024
        self.hidden_size = 768

        self.return_text_representation = False

        # 全参微调时可以使用
        self.learning_rate = 2e-5

        # LoRA 微调时可以使用
        self.lora_learning_rate = 5e-3
        self.fc_learning_rate = 1e-3

        self.lora_weight_decay = 0.0
        self.fc_weight_decay = 0.01

        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-8

        self.scheduler_milestones = [30]
        self.scheduler_gamma = 0.1

        self.max_grad_norm = 1.0


class RoBERTa_Model(nn.Module):

    def __init__(self, data_config, model_config):
        super(RoBERTa_Model, self).__init__()

        self.data_config = data_config
        self.model_config = model_config

        self.bert = AutoModel.from_pretrained(self.model_config.bert_path)

        # 全参微调：RoBERTa 所有参数都可训练
        for param in self.bert.parameters():
            param.requires_grad = True

        hidden_size = getattr(
            self.bert.config,
            "hidden_size",
            self.model_config.hidden_size
        )

        self.fc = nn.Linear(hidden_size, self.data_config.num_classes)

    def forward(self, x, mask):
        output = self.bert(
            input_ids=x,
            attention_mask=mask,
            return_dict=True
        )

        text_representation = output.last_hidden_state[:, 0, :]

        out = self.fc(text_representation)

        if self.model_config.return_text_representation:
            return text_representation, out
        else:
            return out


class LoRALinear(nn.Module):

    def __init__(self, base_linear, r=8, alpha=16, dropout=0.1):
        super(LoRALinear, self).__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear 必须是 nn.Linear")

        self.base_linear = base_linear

        # 冻结原始 Linear 参数
        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.dropout = nn.Dropout(dropout)

        # x: [batch, seq_len, in_features]
        # A: [in_features, r]
        # B: [r, out_features]
        self.lora_A = nn.Parameter(
            torch.randn(self.in_features, r) * 0.01
        )

        # B 初始化为 0，保证刚开始 LoRA 不影响原模型输出
        self.lora_B = nn.Parameter(
            torch.zeros(r, self.out_features)
        )

    def forward(self, x):
        base_output = self.base_linear(x)
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B

        return base_output + self.scaling * lora_output


class RoBERTa_LoRA_Model(nn.Module):

    def __init__(self, data_config, model_config):
        super(RoBERTa_LoRA_Model, self).__init__()

        self.data_config = data_config
        self.model_config = model_config

        # 1. 加载 RoBERTa
        self.bert = AutoModel.from_pretrained(self.model_config.bert_path)

        # 2. 冻结整个 RoBERTa 原始参数
        for param in self.bert.parameters():
            param.requires_grad = False

        # 3. LoRA 超参数
        self.lora_r = getattr(self.model_config, "lora_r", 8)
        self.lora_alpha = getattr(self.model_config, "lora_alpha", 16)
        self.lora_dropout = getattr(self.model_config, "lora_dropout", 0.1)

        # 4. RoBERTa 是 BERT-like 结构，默认只对 query/value 插入标准 LoRA
        self.lora_target_modules = getattr(
            self.model_config,
            "lora_target_modules",
            (
                "attention.self.query",
                "attention.self.value"
            )
        )

        # 5. 注入 LoRA
        self.inject_lora_to_roberta(self.bert)

        # 6. 获取 hidden size
        hidden_size = getattr(
            self.bert.config,
            "hidden_size",
            self.model_config.hidden_size
        )

        # 7. 分类头，默认可训练
        self.fc = nn.Linear(hidden_size, self.data_config.num_classes)

        # 8. 打印可训练参数
        self.print_trainable_parameters()

    def inject_lora_to_roberta(self, module, prefix=""):

        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear) and self.is_lora_target(full_name):
                setattr(
                    module,
                    name,
                    LoRALinear(
                        base_linear=child,
                        r=self.lora_r,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout
                    )
                )
            else:
                self.inject_lora_to_roberta(child, full_name)

    def is_lora_target(self, full_name):

        if "pooler" in full_name:
            return False

        for target in self.lora_target_modules:
            if full_name.endswith(target):
                return True

        return False

    def forward(self, x, mask):
        output = self.bert(
            input_ids=x,
            attention_mask=mask,
            return_dict=True
        )

        # RoBERTa 通常取第一个 token <s> 的表示做分类
        text_representation = output.last_hidden_state[:, 0, :]

        out = self.fc(text_representation)

        if self.model_config.return_text_representation:
            return text_representation, out
        else:
            return out

    def print_trainable_parameters(self):
        trainable_params = 0
        total_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()

            if param.requires_grad:
                trainable_params += param.numel()
                print("Trainable:", name, param.shape)

        print(
            f"Trainable params: {trainable_params} | "
            f"Total params: {total_params} | "
            f"Trainable ratio: {100 * trainable_params / total_params:.4f}%"
        )