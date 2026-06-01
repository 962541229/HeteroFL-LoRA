# coding: UTF-8
from torch.optim import AdamW
from torch.optim import lr_scheduler


def build_svf_lora_optimizer(model, model_config):
    svf_params = []
    fc_decay_params = []
    fc_no_decay_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "delta_sigma" in name:
            svf_params.append(param)
        elif name.startswith("fc."):
            if "bias" in name:
                fc_no_decay_params.append(param)
            else:
                fc_decay_params.append(param)
        else:
            other_params.append(param)
            print("Warning: other trainable parameter:", name, param.shape)

    groups = []
    if svf_params:
        groups.append({"params": svf_params, "lr": model_config.svf_learning_rate, "weight_decay": model_config.svf_weight_decay})
    if fc_decay_params:
        groups.append({"params": fc_decay_params, "lr": model_config.fc_learning_rate, "weight_decay": model_config.fc_weight_decay})
    if fc_no_decay_params:
        groups.append({"params": fc_no_decay_params, "lr": model_config.fc_learning_rate, "weight_decay": 0.0})
    if other_params:
        groups.append({"params": other_params, "lr": model_config.fc_learning_rate, "weight_decay": 0.0})
    if not groups:
        raise RuntimeError("No trainable parameters were found.")

    optimizer = AdamW(groups, betas=(model_config.adam_beta1, model_config.adam_beta2), eps=model_config.adam_eps)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=model_config.scheduler_milestones, gamma=model_config.scheduler_gamma)
    return optimizer, scheduler
