export CUDA_VISIBLE_DEVICES=0

model_family="qwen2-1.5b" # qwen2-1.5b, llama2-7b
use_lora="noLORA" # LORA, noLORA
lora_r=0 # 8, 0
ft_lr="1.5e-05"
ft_epochs=5
ft_wd=0.01
retain_split="full"
FORGET_SPLITS=("forget01_perturbed") # ("forget01_perturbed" "forget05_perturbed" "forget10_perturbed")
checkpoint="checkpoint-625" # 625 for ebs of 32, 1250 for ebs of 16, etc.
seed=42

for split_idx in "${!FORGET_SPLITS[@]}"
do
    forget_split=${FORGET_SPLITS[$split_idx]}

    # Finetune
    retain_path="models/${model_family}_ft_tofu_${use_lora}_epochs${ft_epochs}_lr${ft_lr}_wd${ft_wd}_${retain_split}_seed${seed}"
    torchrun finetune.py --config-name=finetune.yaml \
        model_family=${model_family} \
        use_lora=${use_lora} \
        LoRA.r=${lora_r} \
        batch_size=16 \
        gradient_accumulation_steps=2 \
        lr=${ft_lr} \
        weight_decay=${ft_wd} \
        num_epochs=${ft_epochs} \
        split=${retain_split} \
        seed=${seed} \
        save_dir=${retain_path}

    # Evaluate
    retain_path="${retain_path}/${checkpoint}"

    # retain_path="models/qwen2-1.5b"

    torchrun evaluate_util.py --config-name=eval_everything.yaml \
        model_family=${model_family} \
        model_path=${retain_path} \
        split=${forget_split} \
        batch_size=32


    # Aggregate stats
    python3 aggregate_eval_stat.py \
        retain_result="${retain_path}/eval_results/ds_size300/eval_log_aggregated.json" \
        ckpt_result="${retain_path}/eval_results/ds_size300/eval_log_aggregated.json" \
        method_name="retain-model_${model_family}_ft_tofu_${use_lora}_epochs${ft_epochs}_lr${ft_lr}_wd${ft_wd}_${retain_split}_seed${seed}" \
        save_file="results/${model_family}_ft_DS_${use_lora}_lr${ft_lr}_${checkpoint}_${retain_split}_${forget_split}.csv"
done
