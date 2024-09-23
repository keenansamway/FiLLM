from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
import torch
# from bitsandbytes.optim import AdamW
# from torch.optim import AdamW

from optim.sophia import SophiaG
from optim.datainf import DataInfOptimizer
from optim.sophia_local import SophiaLocal


def get_decay_parameter_names(model):
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def create_adamw_optimizer(model, lr, betas, weight_decay, optim_bits=32, is_paged=True):
    """
    ref1: https://huggingface.co/docs/bitsandbytes/main/en/reference/optim/adamw#bitsandbytes.optim.AdamW
    ref1: https://github.com/huggingface/transformers/blob/fd06ad5438249a055d0b2fd2fc2567d8265a7e4b/src/transformers/trainer.py#L1235
    """
    decay_parameters = get_decay_parameter_names(model)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "lr": lr,
        "betas": betas,
        "weight_decay": weight_decay
    }

    if is_paged:
        optimizer_kwargs["optim_bits"] = optim_bits
        optimizer_kwargs["is_paged"] = True
        from bitsandbytes.optim import AdamW # type: ignore
    else:
        from torch.optim import AdamW

    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

    return optimizer


def create_sophia_optimizer(model, weight_decay, lr, betas, rho):
    decay_parameters = get_decay_parameter_names(model)
    optimizer_grouped_parameters = [
        {
            "params": [p for n,p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n,p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "lr": lr,
        "betas": betas,
        "rho": rho,
        "weight_decay": weight_decay,
    }

    optimizer = SophiaG(optimizer_grouped_parameters, **optimizer_kwargs)

    return optimizer


def create_sophialocal_optimizer(model, weight_decay, lr, betas, rho):
    decay_parameters = get_decay_parameter_names(model)
    optimizer_grouped_parameters = {}
    optimizer_grouped_parameters[0] = [
        {
            "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]
    optimizer_grouped_parameters[1] = [
        {
            "params": [(n,p) for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": weight_decay,
        },
        {
            "params": [(n,p) for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "lr": lr,
        "betas": betas,
        "rho": rho,
        "weight_decay": weight_decay,
    }

    optimizer = SophiaLocal(optimizer_grouped_parameters, **optimizer_kwargs)

    return optimizer

def create_datainf_optimizer(model, lr, lam, weight_decay, retain_dataset, forget_dataset, collate_fn):
    decay_parameters = get_decay_parameter_names(model)
    optimizer_grouped_parameters = {}
    optimizer_grouped_parameters[0] = [
        {
            "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]
    optimizer_grouped_parameters[1] = [
        {
            "params": [(n,p) for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": weight_decay,
        },
        {
            "params": [(n,p) for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = DataInfOptimizer(model,
                                 (retain_dataset, forget_dataset, collate_fn),
                                 optimizer_grouped_parameters,
                                 lr=lr,
                                 lam=lam)

    return optimizer
