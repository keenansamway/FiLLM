"""
Implementation of DataInf
Ref: https://arxiv.org/abs/2310.00902
"""
import math
import os
import numpy as np
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional
from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


from utils import get_datapoint_hash
from localization.RapidGrad import RapidGrad


TQDM_DISABLE = False

def compute_gradients(model, dataset, collate_fn, batch_size=1, compressed_size=None, cache_compressed_grads=False):
    """
    Compute the gradients of the loss function for the training data.
    """
    oporp_eng = RapidGrad(config=None, map_location="cuda")
    dataloader_stochastic = DataLoader(dataset,
                                              shuffle=False,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)
    device = model.device
    model.eval()
    dl_grad_dict = []
    for datapoint_i, batch in enumerate(tqdm(dataloader_stochastic, disable=TQDM_DISABLE)):
        datapoint_hash = get_datapoint_hash(batch['labels'])

        model.zero_grad() # zeroing out gradient

        # batch['labels'] = batch['input_ids'] # DataInf does this but idk why

        # move data to same device as model
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        grad_dict = {}
        for k,v in model.named_parameters():
            grad = v.grad
            if len(v.shape) == 1:
                grad = grad.unsqueeze(1)
            grad = grad.reshape(-1)

            if compressed_size is not None and len(grad) > compressed_size:
                if cache_compressed_grads:
                    grads_dir = os.path.join("compressed_grads", str(compressed_size), model.config.architectures[0], k)
                    os.makedirs(grads_dir, exist_ok=True)
                    grad_filename = os.path.join(grads_dir, f"{datapoint_hash}.pt")
                    if os.path.exists(grad_filename):
                        grad = torch.load(grad_filename, weights_only=True, map_location=device)
                    else:
                        grad = oporp_eng(grad, compressed_size)
                        if os.path.exists(grad_filename):
                            raise Warning(f"Grad file {grad_filename} already exists")
                        else:
                            torch.save(grad, grad_filename)
                else:
                    grad = oporp_eng(grad, compressed_size)
            grad_dict[k] = grad
            del grad

        dl_grad_dict.append(grad_dict)
        del grad_dict
    return dl_grad_dict


def second_order_influential_params(model, data, lam, compressed_size=None, cache_compressed_grads=True):
    # compute the gradients for the retain and unlearn datasets for the layer
    retain_dataset, forget_dataset, collate_fn = data
    retain_grads = compute_gradients(model, retain_dataset, collate_fn, 1, compressed_size, cache_compressed_grads)
    forget_grads = compute_gradients(model, forget_dataset, collate_fn, 1, compressed_size, cache_compressed_grads)

    n_retain = len(retain_grads)
    n_forget = len(forget_grads)

    influence_updates = []

    for param_name, param in tqdm(model.named_parameters(), disable=TQDM_DISABLE):

        # lambda_l computation (lam * mean of the squared norms of the gradients from Grosse et al., 2021)
        S = torch.zeros(n_retain)
        for retain_id in range(n_retain):
            tmp_grad = retain_grads[retain_id][param_name]
            S[retain_id] = torch.mean(tmp_grad**2)
        lambda_l = lam * torch.mean(S)
        # lambda_l = lam

        # Average across all grads in the forget dataset
        tmp_for_grad = forget_grads[0][param_name]
        for id in range(1,n_forget):
            tmp_for_grad += forget_grads[id][param_name]
        tmp_for_grad /= n_forget
        tmp_for_grad = tmp_for_grad.reshape(-1, 1)

        # influence computation
        hvp = torch.zeros(retain_grads[0][param_name].shape)
        hvp = hvp.to(retain_grads[0][param_name].device)

        for retain_id in range(n_retain):
            tmp_ret_grad = retain_grads[retain_id][param_name].reshape(-1, 1)

            numerator = torch.matmul(tmp_ret_grad.T, tmp_for_grad)
            numerator = torch.matmul(tmp_ret_grad, numerator)

            denominator = lambda_l + torch.sum(tmp_ret_grad**2)

            hvp += (tmp_for_grad - torch.nan_to_num(numerator / denominator)).squeeze()

        update = torch.nan_to_num(- 1/(n_retain * lambda_l) * hvp)

        # update = torch.abs(update).mean().item()
        # update = torch.norm(update, p=1).item() # L1-norm
        update = torch.norm(update, p=2).item() # L2-norm

        influence_updates.append((param_name, update))

    sorted_influential_params = [k for (k,v) in sorted(influence_updates, key=lambda x:x[1], reverse=True)]

    return sorted_influential_params

def second_order_influential_params_sum(model, data, lam, compressed_size=None, cache_compressed_grads=True):
    # compute the gradients for the retain and unlearn datasets for the layer
    retain_dataset, forget_dataset, collate_fn = data
    retain_grads = compute_gradients_sum(model, retain_dataset, collate_fn, 1, compressed_size, cache_compressed_grads)
    forget_grads = compute_gradients_sum(model, forget_dataset, collate_fn, 1, compressed_size, cache_compressed_grads)

    n_retain = len(retain_grads)
    n_forget = len(forget_grads)

    influence_updates = []

    for param_name, param in tqdm(model.named_parameters(), disable=TQDM_DISABLE):

        lambda_l = lam

        tmp_for_grad = forget_grads[param_name].reshape(-1, 1)
        tmp_ret_grad = retain_grads[param_name].reshape(-1, 1)

        numerator = torch.matmul(tmp_ret_grad.T, tmp_for_grad)
        numerator = torch.matmul(tmp_ret_grad, numerator)

        denominator = lambda_l + torch.sum(tmp_ret_grad**2)

        hvp = (tmp_for_grad - torch.nan_to_num(numerator / denominator)).squeeze()

        update = torch.nan_to_num(- 1/(n_retain * lambda_l) * hvp)

        update = torch.abs(update).mean().item()

        # update = torch.norm(update, p=1).item() # L1-norm
        # update = torch.norm(update, p=2).item() # L2-norm

        influence_updates.append((param_name, update))

    sorted_influential_params = [k for (k,v) in sorted(influence_updates, key=lambda x:x[1], reverse=True)]

    return sorted_influential_params


def second_order_influential_params_pd(model, data, lam, compressed_size=None, cache_compressed_grads=True):
    # compute the gradients for the retain and unlearn datasets for the layer
    retain_dataset, forget_dataset, collate_fn = data
    retain_grads = compute_gradients(model, retain_dataset, collate_fn, 1, compressed_size, cache_compressed_grads)
    forget_grads = compute_gradients(model, forget_dataset, collate_fn, 1, compressed_size, cache_compressed_grads)

    n_retain = len(retain_grads)
    n_forget = len(forget_grads)

    influence_updates_per_datapoint = []

    for id in tqdm(range(n_forget), disable=TQDM_DISABLE):

        influence_updates = []

        for param_name, param in tqdm(model.named_parameters(), disable=True):

            # lambda_l computation (lam * mean of the squared norms of the gradients from Grosse et al., 2021)
            S = torch.zeros(n_retain)
            for retain_id in range(n_retain):
                tmp_grad = retain_grads[retain_id][param_name]
                S[retain_id] = torch.mean(tmp_grad**2)
            lambda_l = lam * torch.mean(S)

            # Get grad for point in forget dataset
            tmp_for_grad = forget_grads[id][param_name]
            tmp_for_grad = tmp_for_grad.reshape(-1, 1)

            # influence computation
            hvp = torch.zeros(retain_grads[0][param_name].shape)
            hvp = hvp.to(retain_grads[0][param_name].device)

            for retain_id in range(n_retain):
                tmp_ret_grad = retain_grads[retain_id][param_name].reshape(-1, 1)

                numerator = torch.matmul(tmp_ret_grad.T, tmp_for_grad)
                numerator = torch.matmul(tmp_ret_grad, numerator)

                denominator = lambda_l + torch.sum(tmp_ret_grad**2)

                hvp += (tmp_for_grad - torch.nan_to_num(numerator / denominator)).squeeze()

            update = torch.nan_to_num(- 1/(n_retain * lambda_l) * hvp)

            update = torch.abs(update).mean().item()
            # update = torch.norm(update, p=1).item() # L1-norm
            # update = torch.norm(update, p=2).item() # L2-norm

            influence_updates.append((param_name, update))

        sorted_influential_params = [k for (k,v) in sorted(influence_updates, key=lambda x:x[1], reverse=True)]
        influence_updates_per_datapoint.append(sorted_influential_params)

    return influence_updates_per_datapoint


def first_order_influential_params(model, data, lam=None, compressed_size=None, cache_compressed_grads=True):
    # compute the gradients for the retain and unlearn datasets for the layer
    retain_dataset, forget_dataset, collate_fn = data
    retain_grads = compute_gradients(model, retain_dataset, collate_fn, 1, compressed_size, cache_compressed_grads)
    forget_grads = compute_gradients(model, forget_dataset, collate_fn, 1, compressed_size, cache_compressed_grads)

    n_retain = len(retain_grads)
    n_forget = len(forget_grads)

    forget_point_influence = defaultdict(lambda: 0.0)

    # for forget_id in tqdm(range(n_forget), disable=TQDM_DISABLE):
    #     for retain_id in range(n_retain):
    #         for param_name, param in model.named_parameters():
    #             tmp_for_grad = forget_grads[forget_id][param_name]
    #             tmp_ret_grad = retain_grads[retain_id][param_name]
    #             forget_point_influence[param_name] += torch.abs(torch.matmul(tmp_for_grad, tmp_ret_grad)).item()

    # batch code
    # Define chunk size
    chunk_size = 100  # Adjust this based on your GPU memory

    forget_point_influence = defaultdict(lambda: 0.0)

    for param_name, param in tqdm(model.named_parameters(), disable=TQDM_DISABLE):
        # Get stacked gradients for the current parameter
        stacked_forget_grads = torch.stack([forget_grads[forget_id][param_name] for forget_id in range(n_forget)])
        stacked_retain_grads = torch.stack([retain_grads[retain_id][param_name] for retain_id in range(n_retain)])

        # Process in chunks to avoid memory overflow
        for i in range(0, n_forget, chunk_size):
            for j in range(0, n_retain, chunk_size):
                forget_chunk = stacked_forget_grads[i:i+chunk_size]
                retain_chunk = stacked_retain_grads[j:j+chunk_size]

                # Perform batch matrix multiplication
                matmul_results = torch.abs(torch.matmul(forget_chunk, retain_chunk.T))

                # Sum across the retain_id axis and add to the influence dictionary
                forget_point_influence[param_name] += matmul_results.sum(dim=1).sum().item()

    # Average the influence of each forget point
    for k in forget_point_influence.keys():
        forget_point_influence[k] /= ( n_forget)

    sorted_influential_params = [k for (k,v) in sorted(forget_point_influence.items(), key=lambda x:x[1], reverse=True)]

    return sorted_influential_params


def gradient_influential_params(model, data, lam=None, compressed_size=None, cache_compressed_grads=True):
    # compute the gradients for the retain and unlearn datasets for the layer
    retain_dataset, forget_dataset, collate_fn = data
    # retain_grads = compute_gradients(model, retain_dataset, collate_fn, 1, compressed_size)
    forget_grads = compute_gradients(model, forget_dataset, collate_fn, 1, compressed_size, cache_compressed_grads)

    # n_retain = len(retain_grads)
    n_forget = len(forget_grads)

    forget_point_influence = defaultdict(lambda: 0.0)

    # sum the gradients for each parameter across all forget points
    for param_name, param in tqdm(model.named_parameters(), disable=TQDM_DISABLE):
        for forget_id in range(n_forget):
            tmp_for_grad = forget_grads[forget_id][param_name]
            forget_point_influence[param_name] += torch.abs(tmp_for_grad).sum().item()

    # Average the influence of each forget point
    for k in forget_point_influence.keys():
        forget_point_influence[k] /= ( n_forget)

    sorted_influential_params = [k for (k,v) in sorted(forget_point_influence.items(), key=lambda x:x[1], reverse=True)]

    return sorted_influential_params


def memflex(model, data, lam=None, compressed_size=None, cache_compressed_grads=True):
    # compute the gradients for the retain and unlearn datasets for the layer
    retain_dataset, forget_dataset, collate_fn = data
    grad_retention = compute_gradients_sum(model, retain_dataset, collate_fn, 1, compressed_size)
    grad_unlearn = compute_gradients_sum(model, forget_dataset, collate_fn, 1, compressed_size, cache_compressed_grads)

    delta_matrix = {}
    unlearn_list = []
    item_list = []

    for k, _ in grad_unlearn.items():
        if k in grad_retention:
            delta_matrix[k] = compute_cosine_similarity(grad_unlearn[k], grad_retention[k]).squeeze()
            num_unlearn = torch.mean(torch.abs(grad_unlearn[k]))
            unlearn_list.append(num_unlearn)
            item_list.append(delta_matrix[k])

    sim_thre = 0.22 # 0.92 # mu
    grad_thre = 4.79e-5 # 6e-4 # sigma

    item_array = torch.stack(item_list).cpu().float().numpy()
    unlearn_array = torch.stack(unlearn_list).cpu().float().numpy()
    unlearn_sim_idx = np.where(item_array < sim_thre)[0]
    unlearn_grad_idx = np.where(unlearn_array > grad_thre)[0]

    located_region_num = list(np.intersect1d(unlearn_sim_idx, unlearn_grad_idx))
    located_region = []
    for i, key in enumerate(grad_unlearn.keys()):
        if i in located_region_num:
            located_region.append(key)

    return located_region


def compute_cosine_similarity(p, q):
    # p = p.numpy()
    # q = q.numpy()
    p = p.reshape(1, -1)
    q = q.reshape(1, -1)
    return torch.nn.functional.cosine_similarity(p, q)


def compute_gradients_sum(model, dataset, collate_fn, batch_size=1, compressed_size=None, cache_compressed_grads=False):
    """
    Compute the gradients of the loss function for the training data.
    """
    oporp_eng = RapidGrad(config=None, map_location="cuda")
    dataloader_stochastic = DataLoader(dataset,
                                              shuffle=False,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)
    device = model.device
    model.eval()
    grad_dict = {}
    for name, param in model.named_parameters():
        grad_dict[name] = torch.zeros_like(param, device=device)
    for datapoint_i, batch in enumerate(tqdm(dataloader_stochastic, disable=TQDM_DISABLE)):
        # datapoint_hash = get_datapoint_hash(batch['labels'])

        model.zero_grad() # zeroing out gradient

        # batch['labels'] = batch['input_ids'] # DataInf does this but idk why

        # move data to same device as model
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        for k,v in model.named_parameters():
            grad = v.grad
            grad_dict[k] += grad.detach()

    for k in grad_dict.keys():
        grad_dict[k] /= len(dataset)

    return grad_dict