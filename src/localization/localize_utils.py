import os
import random
import torch
from tqdm import tqdm

from localization.influence import (
    second_order_influential_params,
    second_order_influential_params_pd,
    second_order_influential_params_sum,
    first_order_influential_params,
    gradient_influential_params,
    memflex
)
from data_modules.tofu import TOFU_RetainDatasetQA, TOFU_ForgetDatasetQA, datainf_collater
from utils import get_datapoint_hash


def get_ranked_params(model, cfg, tokenizer, max_length, save_ranking=True, cache_compressed_grads=False):
    """
    Computes the most influential parameters based on the method specified in the config.
    """
    local_fn =  gradient_influential_params if cfg.local.method == "gradient" else \
                first_order_influential_params if cfg.local.method == "fo-influence" else \
                second_order_influential_params if cfg.local.method == "so-influence" else \
                second_order_influential_params_sum if cfg.local.method == "so-influence-sum" else \
                memflex if cfg.local.method == "memflex" else None

    compressed_size = 2**cfg.local.compression_power

    # num_ret_samples = 3960 if cfg.data.split == "forget01" else 3800 if cfg.data.split == "forget05" else 3600
    num_ret_samples = cfg.local.num_retain
    num_for_samples = 40 if cfg.data.split == "forget01" else 200 if cfg.data.split == "forget05" else 400
    ranking_filename = f"{cfg.model_path}/{cfg.local.method}_ranked_params_{num_ret_samples}r{num_for_samples}f_{compressed_size}.txt"

    if os.path.exists(ranking_filename):
        with open(ranking_filename, "r") as f:
            influential_params = f.read().split("\n")
        print("Loaded sorted influential params.")
    else:
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        retain_dataset = TOFU_RetainDatasetQA(cfg.data.path, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.data.split, indices=range(num_ret_samples))
        forget_dataset = TOFU_ForgetDatasetQA(cfg.data.path, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.data.split, indices=None)
        influential_params = local_fn(model, (retain_dataset, forget_dataset, datainf_collater), lam=cfg.lam, compressed_size=compressed_size, cache_compressed_grads=cache_compressed_grads)

        # save list of most influential params
        if save_ranking:
            with open(ranking_filename, "w") as f:
                f.write("\n".join(influential_params))
            print("Saved sorted influential params.")

    torch.cuda.empty_cache()
    return influential_params

def get_ranked_params_pd(model, cfg, tokenizer, max_length, save_ranking=True, cache_compressed_grads=False):
    """
    Computes the most influential parameters per datapoint based on the method specified in the config.
    """
    local_fn = second_order_influential_params_pd
    compressed_size = 2**cfg.local.compression_power

    # num_ret_samples = 3960 if cfg.data.split == "forget01" else 3800 if cfg.data.split == "forget05" else 3600
    num_ret_samples = cfg.local.num_retain
    num_for_samples = 40 if cfg.data.split == "forget01" else 200 if cfg.data.split == "forget05" else 400

    ranking_filename = f"{cfg.model_path}/ranked-params-pd_{cfg.local.method}_{num_ret_samples}r{num_for_samples}f_{compressed_size}.txt"

    if os.path.exists(ranking_filename):
        with open(ranking_filename, "r") as f:
            influential_params_pd = [line.split(",") for line in f.read().split("\n") if line]
        print("Loaded sorted influential params per datapoint.")

    else:
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        retain_dataset = TOFU_RetainDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.data.split, indices=range(num_ret_samples), shuffle=False)
        forget_dataset = TOFU_ForgetDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.data.split, indices=None, shuffle=False)
        influential_params_pd = local_fn(model,
                                        (retain_dataset, forget_dataset, datainf_collater),
                                        lam=cfg.lam,
                                        compressed_size=compressed_size,
                                        cache_compressed_grads=cache_compressed_grads)
        if save_ranking:
            # save the list of lists of ranked params
            with open(ranking_filename, "w") as f:
                for params in influential_params_pd:
                    f.write(",".join(params))
                    f.write("\n")
            print("Saved sorted influential params per datapoint.")

    # compute datapoint hashes
    hashes = []
    forget_dataset = TOFU_ForgetDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.data.split, indices=None, shuffle=False)
    for i in range(len(forget_dataset)):
        datapoint_hash = get_datapoint_hash(forget_dataset[i][1])
        hashes.append(datapoint_hash)

    torch.cuda.empty_cache()
    return zip(hashes, influential_params_pd)

def param_subset_selection(params, in_scope, out_scope=['']):
    """
    Selects a subset of the parameters based on the scope list (preserves order).
    """
    subset = [p for p in params for layer_name in in_scope if layer_name in p]
    if out_scope != ['']:
        subset = [p for p in subset if not any([o in p for o in out_scope])]
    return subset

def param_shuffle(params, seed=42):
    """
    Shuffles the parameters based on the seed.
    """
    random.seed(seed)
    random.shuffle(params)

def k_subset_selection(params, k):
    """
    Selects a subset of the parameters based on the value of k.
    If k is a positive value, the top-k% of the parameters are selected.
    If k is a negative value, the bottom-k% of the parameters are selected.
    """
    local_k = round((k)*len(params))
    if 1 > k > 0:
        params = params[:max(1,local_k)]
    elif -1 < k < 0:
        params = params[min(-1, local_k):]
    return params

def k_subset_selection_proportional(params, k, k_offset=0):
    """
    Selects a subset of the parameters based on the value of proportionally to the number of parameters in each category.
    """
    mlp_params = [p for p in params if "mlp" in p]
    attn_params = [p for p in params if "attn" in p]
    norm_params = [p for p in params if "norm" in p]
    embed_param = [p for p in params if "embed" in p]

    mlp_params_subset = round(len(mlp_params)*k) + round(len(mlp_params)*k_offset)
    attn_params_subset = round(len(attn_params)*k) + round(len(mlp_params)*k_offset)
    norm_params_subset = round(len(norm_params)*k) + round(len(mlp_params)*k_offset)

    selection = []
    if 1 > k > 0:
        if len(mlp_params) > 0:
            selection += mlp_params[round(len(mlp_params)*k_offset):max(1,mlp_params_subset)]
        if len(attn_params) > 0:
            selection += attn_params[round(len(mlp_params)*k_offset):max(1,attn_params_subset)]
        if len(norm_params) > 0:
            selection += norm_params[round(len(mlp_params)*k_offset):max(1,norm_params_subset)]
    elif -1 < k < 0:
        if len(mlp_params) > 0:
            selection += mlp_params[min(-1, mlp_params_subset):]
        if len(attn_params) > 0:
            selection += attn_params[min(-1, attn_params_subset):]
        if len(norm_params) > 0:
            selection += norm_params[min(-1, norm_params_subset):]

    selection += embed_param
    return selection

def freeze_other_params(model, params):
    """
    Freezes all the parameters except the ones in the params list.
    """
    for name, param in model.named_parameters():
        if name in params:
            param.requires_grad = True
        else:
            param.requires_grad = False