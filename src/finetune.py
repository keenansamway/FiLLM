import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
import hydra
import transformers
import os
from peft import LoraConfig, get_peft_model
from pathlib import Path
from omegaconf import OmegaConf

from utils import get_model_identifiers_from_yaml
from data_modules import (
    TOFU_TextDatasetQA, TOFU_data_collator,
    KnowUnDo_TextDatasetQA, KnowUnDo_data_collator
)
from trainer import CustomTrainer


@hydra.main(version_base=None, config_path="config", config_name="finetune")
def main(cfg):
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    # save the cfg file
    #if master process
    if os.environ.get('LOCAL_RANK') is None or local_rank == 0:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 500
    if "tofu" in cfg.data.name:
        if "-" in cfg.data.name:
            num_ft_points = int((cfg.data.name).split("-")[1])
            num_ft_points = -1 if num_ft_points == 4000 else num_ft_points
        else:
            num_ft_points = -1
        torch_format_dataset = TOFU_TextDatasetQA(cfg.data.path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.data.split, num_ft_points=num_ft_points)
        dataset_collator = TOFU_data_collator
    elif "knowundo" in cfg.data.name:
        data_type = (cfg.data.name).split("-")[1]
        torch_format_dataset = KnowUnDo_TextDatasetQA(cfg.data.path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.data.split, data_type=data_type)
        dataset_collator = KnowUnDo_data_collator
    else:
        raise ValueError(f"Unknown dataset {cfg.data.name}")

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    # --nproc_per_node gives the number of GPUs per = num_devices. take it from torchrun/os.environ
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    if cfg.is_paged:
        optimizer_type = "paged_adamw_32bit"
    else:
        optimizer_type = "adamw_torch"

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    save_steps = cfg.save_steps if cfg.save_steps > 0 else max_steps
    # max_steps=5
    print(f"max_steps: {max_steps}")
    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # warmup_steps=max(1, max_steps//10),
            warmup_steps=max(1, max_steps//cfg.num_epochs),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim=optimizer_type,
            save_steps=save_steps, #max_steps//4
            save_only_model=True,
            ddp_find_unused_parameters= False,
            eval_strategy="no",
            # deepspeed='src/config/ds_config.json',
            weight_decay = cfg.weight_decay,
            seed = cfg.seed,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)

    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True

    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    if cfg.use_lora == "LORA":
        config = LoraConfig(
            r=cfg.LoRA.r,
            lora_alpha=cfg.LoRA.alpha,
            target_modules="all-linear", #find_all_linear_names(model),
            lora_dropout=cfg.LoRA.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        model.enable_input_require_grads()


    trainer = CustomTrainer(
        model=model,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        args=training_args,
        data_collator=dataset_collator,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    # use model from checkpoints instead, saving here seems to break things when using LoRA and DeepSpeed
    # https://github.com/huggingface/alignment-handbook/issues/57

    # save the model
    # if cfg.use_lora == "LORA":
    #     model = model.merge_and_unload()


    # model.save_pretrained(cfg.save_dir)
    # tokenizer.save_pretrained(cfg.save_dir)

if __name__ == "__main__":
    main()
