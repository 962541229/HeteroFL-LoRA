import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


class DistilBERT_Config(object):
    """模型配置参数"""
    def __init__(self):
        self.model_root_path = "/home/students/wzj_4090_2/code_server/..bert-model/"
        self.model_name = 'distilbert-base-uncased'

        self.save_path = './saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.num_epochs = 60000  # epoch数
        self.batch_size = 64  # mini-batch大小
        self.seq_length = 180  # 每句话处理成的长度(短填长切)
        self.print_per_batch = 1  # 每几个batch输出一次结果

        self.bert_path = self.model_root_path + self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768  # 两个模型，base为768，large为1024

        self.return_text_representation = False

        self.svf_learning_rate = 5e-3
        self.fc_learning_rate = 1e-3

        self.svf_weight_decay = 0.0
        self.fc_weight_decay = 0.01

        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-8

        self.scheduler_milestones = [30]
        self.scheduler_gamma = 0.1

        self.max_grad_norm = 1.0


class DistilBERT_Model(nn.Module):
    def __init__(self, data_config, model_config):
        super(DistilBERT_Model, self).__init__()
        self.data_config = data_config
        self.model_config = model_config

        self.bert = AutoModel.from_pretrained(self.model_config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(self.model_config.hidden_size, self.data_config.num_classes)

    def forward(self, x, mask):
        output = self.bert(x, attention_mask=mask)

        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            text_representation = output.pooler_output
        else:
            text_representation = output.last_hidden_state[:, 0, :]

        out = self.fc(text_representation)

        if self.model_config.return_text_representation:
            return text_representation, out
        else:
            return out


class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16, dropout=0.1):
        super(LoRALinear, self).__init__()

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


        self.lora_A = nn.Parameter(
            torch.randn(self.in_features, r) * 0.01
        )

        self.lora_B = nn.Parameter(
            torch.zeros(r, self.out_features)
        )

    def forward(self, x):
        # 原始 Linear 输出
        base_output = self.base_linear(x)

        # LoRA 增量输出
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B

        return base_output + self.scaling * lora_output


class DistilBERT_LoRA_Model(nn.Module):
    def __init__(self, data_config, model_config):
        super(DistilBERT_LoRA_Model, self).__init__()

        self.data_config = data_config
        self.model_config = model_config

        # 1. 加载 DistilBERT
        self.bert = AutoModel.from_pretrained(self.model_config.bert_path)

        # 2. 冻结整个 DistilBERT 原始参数
        for param in self.bert.parameters():
            param.requires_grad = False

        # 3. LoRA 超参数
        self.lora_r = getattr(self.model_config, "lora_r", 8)
        self.lora_alpha = getattr(self.model_config, "lora_alpha", 16)
        self.lora_dropout = getattr(self.model_config, "lora_dropout", 0.1)

        # DistilBERT attention 中常见线性层：
        # q_lin, k_lin, v_lin, out_lin
        self.lora_target_modules = getattr(
            self.model_config,
            "lora_target_modules",
            ("q_lin", "v_lin")
        )

        # 4. 注入 LoRA
        self.inject_lora_to_distilbert(self.bert)

        # 5. 获取 hidden size
        hidden_size = getattr(
            self.bert.config,
            "hidden_size",
            getattr(self.bert.config, "dim", self.model_config.hidden_size)
        )

        # 6. 分类头，默认可训练
        self.fc = nn.Linear(hidden_size, self.data_config.num_classes)

        # 7. 打印可训练参数
        self.print_trainable_parameters()

    def inject_lora_to_distilbert(self, module):

        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name in self.lora_target_modules:
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
                self.inject_lora_to_distilbert(child)

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