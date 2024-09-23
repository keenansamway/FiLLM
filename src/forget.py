import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
import hydra
import transformers
import os
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from omegaconf import OmegaConf
from collections import defaultdict

# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_modules import (
    TOFU_TextForgetDatasetQA, TOFU_TextForgetDatasetDPOQA, TOFU_data_collator_forget,
    KnowUnDo_TextForgetDatasetQA, KnowUnDo_data_collator_forget,
)
from trainer import CustomTrainerForgetting
from optim import create_adamw_optimizer, create_sophia_optimizer
from utils import get_model_identifiers_from_yaml
from localization.localize_utils import (
    get_ranked_params, get_ranked_params_pd,
    param_subset_selection, param_shuffle,
    k_subset_selection, freeze_other_params,
    k_subset_selection_proportional
)

DDP=False

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    set_seed(cfg.seed)

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")
    # save cfg in cfg.save_dir
    if DDP==False or local_rank == 0:
        if os.path.exists(cfg.save_dir):
            print("Directory already exists")
            if not cfg.overwrite_dir:
                exit()

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 500
    if "tofu" in cfg.data.name:
        if "-" in cfg.data.name:
            num_ft_points = int((cfg.data.name).split("-")[1])
            num_ft_points = -1 if num_ft_points == 4000 else num_ft_points
        else:
            num_ft_points = -1
        if cfg.forget_loss == "dpo":
            torch_format_dataset = TOFU_TextForgetDatasetDPOQA(cfg.data.path, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.data.split)
        else:
            torch_format_dataset = TOFU_TextForgetDatasetQA(cfg.data.path, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.data.split, loss_type=cfg.forget_loss, num_ft_points=num_ft_points)
        dataset_collator = TOFU_data_collator_forget
    elif "knowundo" in cfg.data.name:
        data_type = (cfg.data.name).split("-")[1]
        torch_format_dataset = KnowUnDo_TextForgetDatasetQA(cfg.data.path, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.data.split, data_type=data_type)
        dataset_collator = KnowUnDo_data_collator_forget
    else:
        raise ValueError(f"Unknown dataset {cfg.data.name}")

    if cfg.is_paged:
        optimizer_type = "paged_adamw_32bit"
    else:
        optimizer_type = "adamw_torch"

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    warmup_steps = max(1, steps_per_epoch)
    save_steps = cfg.save_steps if cfg.save_steps != 0 else max_steps
    eval_steps = steps_per_epoch

    print(f"max_steps: {max_steps}, warmup_steps: {warmup_steps}, steps_per_epoch: {steps_per_epoch}, eval_steps: {eval_steps}, save_steps: {save_steps}")

    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim=optimizer_type,
            save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
            save_steps=save_steps,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            # deepspeed='config/ds_config.json',
            weight_decay = cfg.weight_decay,
            eval_steps = eval_steps,
            eval_strategy = "steps" if cfg.eval_while_train else "no",
            seed=cfg.seed
        )

    #first get the base model architectur2e
    #if there is a pytorch*.bin file in the model path, then load that. use regex there can be anythign in between pytorch and .bin
    import re
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search("pytorch.*\.bin", file):
            path_found = True
            break

        if re.search("model-*\.safetensors", file):
            path_found = True
            break

    oracle_model = None

    if path_found:
        config = AutoConfig.from_pretrained(model_id)

        print("Loading from checkpoint")
        # model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, attn_implementation="eager", torch_dtype=torch.bfloat16, trust_remote_code = True)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)
        if cfg.forget_loss == "KL":
            oracle_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)

    else:
        print("Loading after merge and unload")
        model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, device_map=device_map)
        #now use the checkpoint to add the LoRA modules
        model = PeftModel.from_pretrained(model, model_id = cfg.model_path)
        #save this as a standard model so that we can again do PEFT style finetuneing from scratch
        model = model.merge_and_unload()
        #save the model for next time
        model.save_pretrained(cfg.model_path)


    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True

    #now we have a HuggingFace model
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    if cfg.model_family=="phi":
        target_modules=["Wqkv", "out_proj", "fc1", "fc2"]
    elif cfg.model_family=="llama2-7b":
        target_modules=['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj', 'down_proj']
    elif cfg.model_family=="qwen2-1.5b":
        target_modules=['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj', 'down_proj']
    else:
        target_modules=find_all_linear_names(model)

    config = LoraConfig(
        r=cfg.LoRA.r,
        lora_alpha=cfg.LoRA.alpha,
        target_modules="all-linear", #target_modules,
        lora_dropout=cfg.LoRA.dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    if cfg.use_lora == "LORA":
        model = get_peft_model(model, config)
        print_trainable_parameters(model)


    # Determine the most influential parameters

    influence_param_map = defaultdict(list)

    if cfg.local.method == "none":
        model_params = [param_name for param_name, param in model.named_parameters()]

        model_params = param_subset_selection(model_params,
                                              cfg.local.in_scope.split(","),
                                              cfg.local.out_scope.split(","))

         # shuffle the influential params
        if cfg.local.shuffle:
            param_shuffle(model_params, seed=cfg.seed)

        # keep only k% of influential params
        model_params = k_subset_selection(model_params, cfg.local.k)

        with open(f"{cfg.save_dir}/param-scope_{cfg.local.in_scope}_{cfg.local.out_scope}_{cfg.local.shuffle}.txt", "w") as f:
            f.write("\n".join(model_params))

        freeze_other_params(model, model_params)

    elif cfg.local.method in ["gradient", "fo-influence", "so-influence", "memflex", "so-influence-sum"]:
        # get the ranked params
        ranked_params = get_ranked_params(model, cfg, tokenizer, max_length,
                                          save_ranking=True,
                                          cache_compressed_grads=True)

        # select the subset of influential params according to the scopes
        ranked_params = param_subset_selection(ranked_params,
                                               cfg.local.in_scope.split(","),
                                               cfg.local.out_scope.split(","))

        # shuffle the influential params
        if cfg.local.shuffle:
            param_shuffle(ranked_params, seed=cfg.seed)
            param_shuffle(ranked_params, seed=cfg.seed)

        # keep only k% of influential params
        # num_params = round(len(p for p in model.named_parameters()) * cfg.local.k)
        ranked_params = k_subset_selection_proportional(ranked_params, cfg.local.k, cfg.local.k_offset)
        with open(f"{cfg.save_dir}/param-scope_{cfg.local.in_scope}_{cfg.local.out_scope}_{cfg.local.shuffle}_{cfg.local.k}+{cfg.local.k_offset}_{cfg.local.num_retain}r.txt", "w") as f:
            f.write("\n".join(ranked_params))

        freeze_other_params(model, ranked_params)

    elif cfg.local.method == "param-deltas":
        with open(f"{cfg.model_path}/param-deltas_ranked_params.txt", "r") as f:
            ranked_params = f.read().splitlines()

        ranked_params = param_subset_selection(ranked_params,
                                               cfg.local.in_scope.split(","),
                                               cfg.local.out_scope.split(","))

        if cfg.local.shuffle:
            param_shuffle(ranked_params, seed=cfg.seed)

        ranked_params = k_subset_selection_proportional(ranked_params, cfg.local.k, cfg.local.k_offset)
        with open(f"{cfg.save_dir}/param-scope_{cfg.local.in_scope}_{cfg.local.out_scope}_{cfg.local.shuffle}_{cfg.local.k}+{cfg.local.k_offset}.txt", "w") as f:
            f.write("\n".join(ranked_params))

        freeze_other_params(model, ranked_params)

    elif cfg.local.method == "so-influence-pd":
        if cfg.batch_size != 1:
            raise ValueError("Batch size must be 1 for this method.")

        ranked_params_pd = get_ranked_params_pd(model, cfg, tokenizer, max_length,
                                                save_ranking=True,
                                                cache_compressed_grads=True)

        for datapoint_hash, ranked_params in ranked_params_pd:
            # select the subset of influential params according to the scopes
            ranked_params = param_subset_selection(ranked_params,
                                                cfg.local.in_scope.split(","),
                                                cfg.local.out_scope.split(","))

            # shuffle the influential params
            if cfg.local.shuffle:
                param_shuffle(ranked_params, seed=cfg.seed)

            # keep only k% of influential params
            ranked_params = k_subset_selection(ranked_params, cfg.local.k)

            influence_param_map[datapoint_hash] = ranked_params

    elif cfg.local.method == "random":
        model_params = [param_name for param_name, param in model.named_parameters()]

        model_params = param_subset_selection(model_params,
                                              cfg.local.in_scope.split(","),
                                              cfg.local.out_scope.split(","))

        # shuffle the influential params
        if cfg.local.shuffle:
            param_shuffle(model_params, seed=cfg.seed)

        # keep only k% of influential params
        model_params = k_subset_selection(model_params, cfg.local.k)

        with open(f"{cfg.save_dir}/param-scope_random{cfg.seed}.txt", "w") as f:
            f.write("\n".join(model_params))

        freeze_other_params(model, model_params)

    else:
        raise NotImplementedError(f"Local method {cfg.local.method} not implemented")


    # Create the optimizer
    optimizer, lr_scheduler = None, None
    if (cfg.optimizer).lower() == "adam" or (cfg.optimizer).lower() == "adamw":
        optimizer = create_adamw_optimizer(
            model,
            lr=cfg.lr,
            betas=(0.9,0.999),
            weight_decay=cfg.weight_decay,
            optim_bits=32,
            is_paged=cfg.is_paged,
        )
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
        print("Created AdamW optimizer.")
    elif (cfg.optimizer).lower() == "sophiag":
        optimizer = create_sophia_optimizer(
                    model,
                    lr=cfg.lr,
                    betas=(cfg.beta1, cfg.beta2),
                    rho=cfg.rho,
                    weight_decay=cfg.weight_decay,
                )
        print("Created SophiaG optimizer.")
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")

    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        compute_metrics=None,                # the callback for computing metrics, None in this case since you're doing it in your callback
        # callbacks=[GlobalStepDeletionCallback],
        args=training_args,
        data_collator=dataset_collator,
        oracle_model=oracle_model,
        forget_loss=cfg.forget_loss,
        eval_cfg=cfg.eval,
        influence_param_map=influence_param_map,
        optimizers=(optimizer, lr_scheduler),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    # trainer.train()
    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    #save the tokenizer
    # if cfg.save_model and (not cfg.eval_only):
    #     model.save_pretrained(cfg.save_dir)
    #     tokenizer.save_pretrained(cfg.save_dir)

    #delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if DDP==False or local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)



if __name__ == "__main__":
    main()

