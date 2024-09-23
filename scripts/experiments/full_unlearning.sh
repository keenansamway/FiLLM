: '
Script to run experiments comparing different layers for unlearning.
Methods tested
    - LOCAL_LAYERS=("everything" "all" "self-attn" "layernorm")

'
export CUDA_VISIBLE_DEVICES=0

experiment_name="full_unlearn_embed"

# model_family="qwen2-1.5b"
model_family="phi"
use_lora="noLORA"
lora_r=0

# TOFU args #
if [ $1 == "tofu" ]; then
    data_name="tofu-4000"
    data_path="locuslab/TOFU"
    eval_config_path="eval_tofu.yaml"

    ft_epochs=10
    ft_lr="5e-05" # qwen: 2.5e-05
    ft_checkpoint="checkpoint-1250" # tofu: floor(4000 * epochs / batch_size * gradient_accumulation_steps)
    retain_checkpoint="checkpoint-1237" # tofu: floor({3960, 3800, 3600} * epochs / batch_size * gradient_accumulation_steps)

    forget_epochs=8
    FORGET_LR=("1e-05")
    unlearn_split="forget01" # ("forget10" "forget05" "forget01")
    retain_split="retain99" # ("retain90" "retain95" "retain99")
    forget_checkpoint="checkpoint-20" # tofu: floor(40 * epochs / batch_size * gradient_accumulation_steps)
    save_steps=0

# KnowUnDo args #
elif [ $1 == "knowundo" ]; then
    data_name="knowundo-copyright"
    data_path="zjunlp/KnowUnDo"
    eval_config_path="eval_knowundo.yaml"

    ft_epochs=10
    ft_lr="1e-04"
    ft_checkpoint="checkpoint-496" # knowundo: floor(1590 * epochs / batch_size * gradient_accumulation_steps)
    retain_checkpoint="checkpoint-347" # knowundo: floor(1113 * epochs / batch_size * gradient_accumulation_steps)

    forget_epochs=12
    forget_lr="5e-06"
    unlearn_split="unlearn"
    retain_split="retention"
    forget_checkpoint="checkpoint-357" # knowundo: floor(477 * epochs / batch_size * gradient_accumulation_steps)
    save_steps=0

else
    echo "Invalid dataset name."
    exit 1
fi

# Other args args #
ft_wd=0.01
model_path="models/${model_family}_ft_${data_name}_${use_lora}_epochs${ft_epochs}_lr${ft_lr}_wd${ft_wd}_full_seed42/${ft_checkpoint}"

optim="sophiag" # ("adamw" "sophiag")
forget_loss="grad-ascent" # ("grad-ascent" "grad-diff" "KL")

local_method="none" # ("none" "gradient" "fo-influence" "so-influence")
local_num_retain=800
local_k=1
local_in_scope="mlp\,attn\,norm\,embed\,lm_head" # "mlp\,attn\,norm\,embed"
local_out_scope="none"
local_shuffle="False"
compression_power=16

exp=""

for idx in "${!FORGET_LR[@]}"
do
    forget_lr=${FORGET_LR[$idx]}

    # Unlearn
    forget_save_dir="${model_path}/${exp}/${optim}_${forget_loss}_${local_method}_k${local_k}_${forget_lr}_${unlearn_split}_${forget_epochs}"

    torchrun src/forget.py --config-name=forget.yaml \
        model_family=${model_family} \
        model_path=${model_path} \
        use_lora=${use_lora} \
        LoRA.r=${lora_r} \
        batch_size=8 \
        gradient_accumulation_steps=2 \
        data.name=${data_name} \
        data.path=${data_path} \
        data.split=${unlearn_split} \
        num_epochs=${forget_epochs} \
        optimizer=${optim} \
        forget_loss=${forget_loss} \
        lr=${forget_lr} \
        local.method=${local_method} \
        local.k=${local_k} \
        local.num_retain=$local_num_retain \
        local.in_scope=${local_in_scope} \
        local.out_scope=${local_out_scope} \
        local.shuffle=${local_shuffle} \
        local.compression_power=${compression_power} \
        lam=0.1 \
        beta1=0.9 \
        beta2=0.95 \
        rho=0.04 \
        save_dir=${forget_save_dir} \
        overwrite_dir=True \
        save_steps=${save_steps}

    # Evaluate
    eval_forget_save_dir="${forget_save_dir}/${forget_checkpoint}"
    eval_unlearn_split=${unlearn_split}
    if [[ $data_name == *"tofu"* ]]; then
        eval_unlearn_split=${unlearn_split}_perturbed
    fi

    torchrun src/evaluate_util.py --config-name=${eval_config_path} \
        model_family=${model_family} \
        model_path=${eval_forget_save_dir} \
        split=${eval_unlearn_split} \
        batch_size=96

    # Aggregate stats
    retain_path="models/${model_family}_ft_${data_name}_${use_lora}_epochs${ft_epochs}_lr${ft_lr}_wd${ft_wd}_${retain_split}_seed42/${retain_checkpoint}"
    unlearn_info=${eval_unlearn_split}_${forget_epochs}_${optim}_${forget_loss}_${forget_lr}_${local_method}_${local_k}_${local_num_retain}_${local_in_scope}_${local_out_scope}_${local_shuffle}
    python3 src/aggregate_eval_stat.py \
        retain_result="${retain_path}/eval_results/ds_size300/eval_log_aggregated.json" \
        ckpt_result="${eval_forget_save_dir}/eval_results/ds_size300/eval_log_aggregated.json" \
        method_name=${unlearn_info} \
        save_file="results/${model_family}/${data_name}/${experiment_name}/${model_family}_${ft_lr}_${ft_checkpoint}_unlearn_${unlearn_info}.csv"
done
