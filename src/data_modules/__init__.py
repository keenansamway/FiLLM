import torch.nn as nn

from data_modules.tofu import (
    TOFU_TextDatasetQA, TOFU_data_collator,
    TOFU_TextForgetDatasetQA, TOFU_TextForgetDatasetDPOQA, TOFU_data_collator_forget,
    TOFU_data_collator_with_indices
)
from data_modules.knowundo import (
    KnowUnDo_TextDatasetQA, KnowUnDo_data_collator,
    KnowUnDo_TextForgetDatasetQA, KnowUnDo_data_collator_forget,
    KnowUnDo_data_collator_with_indices
)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss