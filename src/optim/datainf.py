"""
Implementation of custom DataInf optimizer for PyTorch.
Ref: https://arxiv.org/abs/2310.00902
"""
import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings


TDQM_DISABLE = False

def compute_influence(model, data, lam, params_to_examine):
    # compute the gradients for the retain and unlearn datasets for the layer
    retain_dataset, forget_dataset, collate_fn = data
    retain_grads = compute_gradients(model, retain_dataset, collate_fn, params_to_examine)
    forget_grads = compute_gradients(model, forget_dataset, collate_fn, params_to_examine)

    n_retain = len(retain_grads)
    n_forget = len(forget_grads)

    influence_updates = []

    for param_name, param in tqdm(params_to_examine, disable=TDQM_DISABLE):

        # lambda_l computation (lam * mean of the squared norms of the gradients from Grosse et al., 2021)
        S = torch.zeros(n_retain)
        for retain_id in range(n_retain):
            tmp_grad = retain_grads[retain_id][param_name]
            S[retain_id] = torch.mean(tmp_grad**2)
        # lambda_l = lam * torch.mean(S)
        lambda_l = lam

        # Average across all grads in the forget dataset
        tmp_for_grad = forget_grads[0][param_name]
        for id in range(1,n_forget):
            tmp_for_grad += forget_grads[id][param_name]
        tmp_for_grad /= n_forget
        tmp_for_grad = tmp_for_grad.reshape(-1, 1)

        # influence computation
        dim1, dim2 = retain_grads[0][param_name].shape
        hvp = torch.zeros((dim1 * dim2))

        for retain_id in range(n_retain):
            tmp_ret_grad = retain_grads[retain_id][param_name].reshape(-1, 1)

            numerator = torch.matmul(tmp_ret_grad.T, tmp_for_grad)
            numerator = torch.matmul(tmp_ret_grad, numerator)

            denominator = lambda_l + torch.sum(tmp_ret_grad**2)

            hvp += (tmp_for_grad - torch.nan_to_num(numerator / denominator)).squeeze()

        update = torch.nan_to_num(- 1/(n_retain * lambda_l) * hvp)
        update = update.reshape(dim1, dim2)
        if len(param.shape) == 1:
            update = update.squeeze()

        influence_updates.append(update)

    return influence_updates


def compute_gradients(model, dataset, collate_fn, params_to_examine, batch_size=1):
    """
    Compute the gradients of the loss function for the training data.

    Per-sample-gradient code adapted from: https://discuss.pytorch.org/t/vectorizing-calculation-of-per-sample-gradient-for-llms/203590
    """
    batch_size = 1
    dataloader_stochastic = DataLoader(dataset,
                                              shuffle=False,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)
    device = model.device
    model.eval()
    dl_grad_dict = []
    for datapoint_i, batch in enumerate(tqdm(dataloader_stochastic, disable=TDQM_DISABLE)):
        model.zero_grad() # zeroing out gradient

        # batch['labels'] = batch['input_ids'] # DataInf does this but idk why

        # move data to same device as model
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        grad_dict = {}
        for k,v in params_to_examine:
            grad = v.grad.cpu()
            if len(v.shape) == 1:
                grad = grad.unsqueeze(1)
            grad_dict[k] = grad

        dl_grad_dict.append(grad_dict)
        del grad_dict
    model.train() # TODO: is this needed?
    return dl_grad_dict


class DataInfOptimizer(Optimizer):
    """
    DataInf optimizer class.

    OG DataInf Steps:
    1. Compute the gradients of the loss function for the train and val data.
    2. Preprocess the gradients.
    3. Compute the HVPs
    4. Compute the Influence Function
    5. Save results

    General steps:
    1. Compute the gradient of the loss function for the training data.
    2. Preprocess the gradients
    3.

    """
    def __init__(self, model, data, params, lr=1e-5, lam=0.1, weight_decay=0.01, *, maximize: bool = False, capturable: bool = False, dynamic: bool = False):
        """
        Initialize the optimizer.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.model = model
        self.data = data

        # manual override for testing
        lr = -1
        weight_decay = 0

        defaults = dict(lr=lr, lam=lam, weight_decay=weight_decay, maximize=maximize, capturable=capturable, dynamic=dynamic)
        super(DataInfOptimizer, self).__init__(params[0], defaults)

        self.param_groups[0]['params'] = params[1][0]['params']
        self.param_groups[1]['params'] = params[1][1]['params']

    def __setstate__(self, state):
        """
        Set the state of the optimizer.
        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
            group.setdefault('dynamic', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    # @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Perform the optimization step
        for group in self.param_groups:
            if len(group['params']) == 0:
                continue    # skip the group if it has no parameters
            params_with_updates = []
            updates = []
            state_steps = []

            # Determine which layers to update
            params_to_examine = []
            # params_per_layer = defaultdict(list)

            for (k,v) in group['params']:
                if v.grad is None:
                    continue
                if v.grad.is_sparse:
                    raise RuntimeError('Sparse gradients are not supported.')
                # if "layers" not in k: # transformer layers only, no embeddings
                #     continue
                # if all([p not in _ for p in [f"{n}.mlp" for n in range(0, 24)]]):
                #     continue
                if 'mlp' not in k:
                    continue
                # if 'lora_A' not in k or 'lora_B' not in k:
                    # continue
                # layer_num = k.split('.')[2]
                # params_per_layer[layer_num].append((k,v))
                params_to_examine.append((k,v))

            if len(params_to_examine) == 0:
                continue

            # Compute the influence update for all layer
            influence_updates = compute_influence(self.model, self.data, group['lam'], params_to_examine)

            for layer, influence_update in zip(params_to_examine, influence_updates):
                params_with_updates.append(layer[1])
                updates.append(influence_update)
                state = self.state[layer[1]]

                # State initialization
                if len(state) == 0:
                    if group['capturable']:
                        state['step'] = torch.zeros((1,), dtype=torch.float, device=layer[1].device)
                    else:
                        state['step'] = torch.tensor(0.)

                state_steps.append(state['step'])

            with torch.no_grad():
                self._datainf(params_with_updates,
                            updates,
                            state_steps,
                            lr=group['initial_lr'],
                            weight_decay=group['weight_decay'],
                            maximize=group['maximize'],
                            capturable=group['capturable'])

        return loss


    def _datainf(self,
                 params: List[Tensor],
                 updates: List[Tensor],
                 state_steps: List[Tensor],
                 capturable: bool = False,
                 *,
                 lr: float,
                 weight_decay: float,
                 maximize: bool,):
        """
        DataInf function.
        """

        # any additional checks

        self._single_tensor_datainf(params,
                                    updates,
                                    state_steps,
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    maximize=maximize,
                                    capturable=capturable)


    def _single_tensor_datainf(self,
                               params: List[Tensor],
                               updates: List[Tensor],
                               state_steps: List[Tensor],
                               *,
                               lr: float,
                               weight_decay: float,
                               maximize: bool,
                               capturable: bool):
        """
        DataInf function for a single tensor.
        """

        for i, (update, param) in enumerate(zip(updates, params)):
            update = update if not maximize else -update
            update = update.to(param.device)
            step_t = state_steps[i]

            if capturable:
                assert param.is_cuda and step_t.is_cuda

            if torch.is_complex(param):
                update = torch.view_as_real(update)
                param = torch.view_as_real(param)

            # update step
            step_t.add_(1)

            # Perform stepweight decay
            # param.mul_(1 - lr * weight_decay)

            if capturable:
                step = step_t
                step_size = lr
                step_size_neg = step_size.neg()
                param.add_(update, alpha=step_size_neg)

            else:
                step = step_t.item()
                step_size_neg = -lr
                param.add_(update, alpha=step_size_neg)

            del update
