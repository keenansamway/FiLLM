import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import BatchEncoding
from datasets import load_dataset, concatenate_datasets

from datasets import logging as datasets_logging
datasets_logging.set_verbosity_error()

from utils import get_model_identifiers_from_yaml, add_dataset_index

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


class TOFU_RetainDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", indices=None, shuffle=True):
        super(TOFU_RetainDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data = load_dataset(data_path, retain_split)["train"]
        self.retain_data = self.retain_data.shuffle(seed=42) if shuffle else self.retain_data
        self.retain_data = self.retain_data.select(indices) if indices is not None else self.retain_data
        self.model_configs = get_model_identifiers_from_yaml(model_family)

    def __len__(self):
        return len(self.retain_data)

    def __getitem__(self, idx):
        question = self.retain_data[idx]['question']
        answer = self.retain_data[idx]['answer']

        converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
        return converted_data

class TOFU_ForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", indices=None, shuffle=True):
        super(TOFU_ForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = load_dataset(data_path, split)["train"]
        self.forget_data = self.forget_data.shuffle(seed=42) if shuffle else self.forget_data
        self.forget_data = self.forget_data.select(indices) if indices is not None else self.forget_data
        self.model_configs = get_model_identifiers_from_yaml(model_family)

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        question = self.forget_data[idx]['question']
        answer = self.forget_data[idx]['answer']

        converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
        return converted_data

def datainf_collater(samples):
    input_ids, labels, attention_masks = zip(*samples)
    # create and return dictionary
    batch = {
        "input_ids": torch.vstack(input_ids),
        "labels": torch.vstack(labels),
        "attention_mask": torch.vstack(attention_masks),
    }
    return BatchEncoding(batch)


class TOFU_TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="idk", num_ft_points=-1):
        super(TOFU_TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = load_dataset(data_path, split)["train"]
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data =load_dataset(data_path, retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if num_ft_points > 0:
            if retain_split == "retain99":
                ret_data1 = self.retain_data.select(range(num_ft_points - 400))
                ret_data2 = self.retain_data.select(range(4000 - (400), 3960))
                self.retain_data = concatenate_datasets([ret_data1, ret_data2])

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            answer = data[idx]['answer']

            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets


class TOFU_TextForgetDatasetDPOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", ):
        super(TOFU_TextForgetDatasetDPOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = load_dataset(data_path, split)["train"]
        self.idontknowfile = "data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data = load_dataset(data_path, retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)


    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)

            question = data[idx]['question']

            if data_type != "idk":
                answer = data[idx]['answer']
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets


class TOFU_TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='question', answer_key='answer', num_ft_points=-1):
        super(TOFU_TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # data_len = len(load_dataset(data_path, split)["train"])
        # self.data = load_dataset(data_path, split)["train"].select(range(min(100, data_len)))
        self.data = load_dataset(data_path, split)["train"]

        self.data = add_dataset_index(self.data)

        if num_ft_points > 0:
            if split == "full":
                ret_data = self.data.select(range(num_ft_points - 400))
                for_data = self.data.select(range(4000 - 400, 4000))
                self.data = concatenate_datasets([ret_data, for_data])

            elif split == "retain99":
                ret_data1 = self.data.select(range(num_ft_points - 400))
                ret_data2 = self.data.select(range(4000 - (400), 3960))
                self.data = concatenate_datasets([ret_data1, ret_data2])

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)


# def TOFU_collate_fn(batch):
#     input_ids, attention_masks = zip(*batch)
#     input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
#     attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
#     return input_ids, attention_masks

def TOFU_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def TOFU_data_collator_forget(samples):
    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

def TOFU_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)
