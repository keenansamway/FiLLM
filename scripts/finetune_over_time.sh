export CUDA_VISIBLE_DEVICES=0

model_family="phi" # qwen2-1.5b
use_lora="noLORA" # LORA, noLORA
lora_r=0 # 8, 0

# TOFU args #
if [ $1 == "tofu" ] || [ $1 == "tofu-4000" ]; then
    data_name="tofu-4000"
    data_path="locuslab/TOFU"
    eval_config_path="eval_tofu.yaml"

    forget_split="forget01_perturbed"
    CHECKPOINT=({125..1250..125}) # knowundo: floor(1590 * epochs / batch_size * gradient_accumulation_steps)
    save_steps=125 # save 10 checkpoints

    FT_LR=("2.5e-05")
    ft_epochs=10

# TOFU args #
elif [ $1 == "tofu-1600" ]; then
    data_name="tofu-1600"
    data_path="locuslab/TOFU"
    eval_config_path="eval_tofu.yaml"

    forget_split="forget01_perturbed"
    CHECKPOINT=({50..500..50}) # knowundo: floor(4000 * epochs / batch_size * gradient_accumulation_steps)
    save_steps=50 # save 10 checkpoints

    FT_LR=("2.5e-05")
    ft_epochs=10

# KnowUnDo args #
elif [ $1 == "knowundo" ]; then
    data_name="knowundo-copyright"
    data_path="zjunlp/KnowUnDo"
    eval_config_path="eval_knowundo.yaml"

    forget_split="unlearn"
    CHECKPOINT=({62..496..62}) # knowundo: floor(1590 * epochs / batch_size * gradient_accumulation_steps)
    save_steps=62 # save 8 checkpoints

    FT_LR=("1e-04")
    ft_epochs=10

else
    echo "Invalid dataset name."
    exit 1
fi

# Other args #
ft_wd=0.01
retain_split="full"
seed=42

# Run experiments #
experiment_name="finetune_over_time_${ft_epochs}"

for ft_lr in "${FT_LR[@]}"
do
    # Finetune
    # retain_path="models/${model_family}_ft_${data_name}_${use_lora}_epochs${ft_epochs}_lr${ft_lr}_wd${ft_wd}_${retain_split}_seed${seed}"
    retain_path="models/${model_family}_ft_${data_name}_${use_lora}_epochs${ft_epochs}_wd${ft_wd}_${retain_split}_seed${seed}"

    torchrun src/finetune.py --config-name=finetune.yaml \
        model_family=${model_family} \
        use_lora=${use_lora} \
        LoRA.r=${lora_r} \
        data.name=${data_name} \
        data.path=${data_path} \
        data.split=${retain_split} \
        batch_size=16 \
        gradient_accumulation_steps=2 \
        lr=${ft_lr} \
        weight_decay=${ft_wd} \
        num_epochs=${ft_epochs} \
        seed=${seed} \
        save_dir=${retain_path} \
        save_steps=${save_steps}

    for idx in "${!CHECKPOINT[@]}"
    do
        # Evaluate
        checkpoint="checkpoint-${CHECKPOINT[$idx]}"
        eval_retain_path="${retain_path}/${checkpoint}"

        torchrun src/evaluate_util.py --config-name=${eval_config_path} \
            model_family=${model_family} \
            model_path=${eval_retain_path} \
            split=${forget_split} \
            batch_size=96 \
            data_name=${data_name}

        # Aggregate stats
        finetune_info=${model_family}_full_${data_name}_${checkpoint}_${use_lora}_${ft_lr}_${ft_epochs}_${ft_wd}_${retain_split}_${forget_split}_seed${seed}
        python3 src/aggregate_eval_stat.py \
            retain_result="${eval_retain_path}/eval_results/ds_size300/eval_log_aggregated.json" \
            ckpt_result="${eval_retain_path}/eval_results/ds_size300/eval_log_aggregated.json" \
            method_name="${finetune_info}" \
            save_file="results/${model_family}/${data_name}/${experiment_name}/${finetune_info}.csv"
    done
done