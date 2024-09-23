export CUDA_VISIBLE_DEVICES=0

# model_family="qwen2-1.5b" # qwen2-1.5b, llama2-7b
model_family="phi"
use_lora="noLORA" # LORA, noLORA
lora_r=0 # 8, 0

# TOFU args #
if [ $1 == "tofu" ] || [ $1 == "tofu-4000" ]; then
    data_name=$1
    data_path="locuslab/TOFU"
    eval_config_path="eval_tofu.yaml"

    RETAIN_SPLITS=("retain99") # ("retain99" "retain95" "retain90")
    FORGET_SPLITS=("forget01_perturbed") # ("forget01_perturbed" "forget05_perturbed" "forget10_perturbed")
    CHECKPOINT=("checkpoint-1237") # ("checkpoint-618" "checkpoint-593" "checkpoint-562")

    ft_lr="5e-05"
    ft_epochs=10

# TOFU args #
elif [ $1 == "tofu-1600" ]; then
    data_name="tofu-1600"
    data_path="locuslab/TOFU"
    eval_config_path="eval_tofu.yaml"

    RETAIN_SPLITS=("retain99") # ("retain99" "retain95" "retain90")
    FORGET_SPLITS=("forget01_perturbed") # ("forget01_perturbed" "forget05_perturbed" "forget10_perturbed")
    CHECKPOINT=("checkpoint-487") # ("checkpoint-618" "checkpoint-593" "checkpoint-562")

    ft_lr="2.5e-05"
    ft_epochs=10

# TOFU args #
elif [ $1 == "tofu-800" ]; then
    data_name="tofu-800"
    data_path="locuslab/TOFU"
    eval_config_path="eval_tofu.yaml"

    RETAIN_SPLITS=("retain99") # ("retain99" "retain95" "retain90")
    FORGET_SPLITS=("forget01_perturbed") # ("forget01_perturbed" "forget05_perturbed" "forget10_perturbed")
    CHECKPOINT=("checkpoint-237") # ("checkpoint-618" "checkpoint-593" "checkpoint-562")

    ft_lr="2.5e-05"
    ft_epochs=10

# KnowUnDo args #
elif [ $1 == "knowundo" ]; then
    data_name="knowundo-copyright"
    data_path="zjunlp/KnowUnDo"
    eval_config_path="eval_knowundo.yaml"

    RETAIN_SPLITS=("retention")
    FORGET_SPLITS=("unlearn")
    CHECKPOINT=("checkpoint-347") # knowundo: floor(1113 * epochs / batch_size * gradient_accumulation_steps)

    ft_lr="1e-04"
    ft_epochs=10

else
    echo "Invalid dataset name."
    exit 1
fi

# Other args #
ft_wd=0.01
seed=42

for split_idx in "${!FORGET_SPLITS[@]}"
do
    retain_split=${RETAIN_SPLITS[$split_idx]}
    forget_split=${FORGET_SPLITS[$split_idx]}
    checkpoint=${CHECKPOINT[$split_idx]}

    # Finetune
    retain_path="models/${model_family}_ft_${data_name}_${use_lora}_epochs${ft_epochs}_lr${ft_lr}_wd${ft_wd}_${retain_split}_seed${seed}"
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
        save_steps=0

    # Evaluate
    eval_retain_path="${retain_path}/${checkpoint}"

    torchrun src/evaluate_util.py --config-name=${eval_config_path} \
        model_family=${model_family} \
        model_path=${eval_retain_path} \
        split=${forget_split} \
        batch_size=96 \
        data_name=${data_name}


    # Aggregate stats
    retain_info="${model_family}_retain_${data_name}_${checkpoint}_${use_lora}_${ft_lr}_${ft_epochs}_${ft_wd}_${retain_split}_${forget_split}_seed${seed}"
    python3 src/aggregate_eval_stat.py \
        retain_result="${eval_retain_path}/eval_results/ds_size300/eval_log_aggregated.json" \
        ckpt_result="${eval_retain_path}/eval_results/ds_size300/eval_log_aggregated.json" \
        method_name=${retain_info} \
        save_file="results/${model_family}/${data_name}/${retain_info}.csv"
done
