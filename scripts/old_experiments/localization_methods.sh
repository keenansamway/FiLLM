: '
Script to run experiments comparing different localization methods for unlearning.
Methods tested
    - LOCAL_METHOD=("none" "gradient" "fo-influence" "so-influence")

'
export CUDA_VISIBLE_DEVICES=0

experiment_name="localization_methods"

model_family="qwen2-1.5b" # qwen2-1.5b, llama2-7b
use_lora="noLORA"
lora_r=0

ft_epochs=5
ft_lr="1.5e-05"
ft_wd=0.01
ft_checkpoint="checkpoint-625" # 625 for ebs of 32, 1250 for ebs of 16, etc.
model_path="models/${model_family}_ft_tofu_${use_lora}_epochs${ft_epochs}_lr${ft_lr}_wd${ft_wd}_full_seed42/${ft_checkpoint}"
retain_checkpoint="checkpoint-618"

forget_checkpoint="checkpoint-12" # ("checkpoint-125" "checkpoint-62" "checkpoint-12") for ebs of 16
unlearn_split="forget01" # ("forget10" "forget05" "forget01")
retain_split="retain99" # ("retain90" "retain95" "retain99")

forget_epochs=5
optim="adamw" # ("adamw" "sophiag")
forget_loss="grad-ascent" # ("grad-ascent" "grad-diff" "KL")
FORGET_LR=("1e-05" "5e-05" "5e-05" "5e-05")

LOCAL_METHOD=("none" "gradient" "fo-influence" "so-influence")
local_num_retain=800
LOCAL_K=(1 0.1 0.1 0.1)
local_in_scope="mlp"
local_out_scope="bias"
local_shuffle="False"
compression_power=16

for idx in "${!LOCAL_METHOD[@]}"
do
    local_method=${LOCAL_METHOD[$idx]}
    local_k=${LOCAL_K[$idx]}
    forget_lr=${FORGET_LR[$idx]}

    # Unlearn
    forget_save_dir="${model_path}/${optim}_${forget_loss}_${local_method}_k${local_k}_${forget_lr}_${unlearn_split}_${forget_epochs}"

    torchrun forget.py --config-name=forget.yaml \
        model_family=${model_family} \
        model_path=${model_path} \
        use_lora=${use_lora} \
        LoRA.r=${lora_r} \
        batch_size=8 \
        gradient_accumulation_steps=2 \
        split=${unlearn_split} \
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
        overwrite_dir=True

    # Evaluate
    forget_save_dir="${forget_save_dir}/${forget_checkpoint}"

    torchrun evaluate_util.py --config-name=eval_everything.yaml \
        model_family=${model_family} \
        model_path=${forget_save_dir} \
        split=${unlearn_split}_perturbed \
        batch_size=32

    # Aggregate stats
    retain_path="models/${model_family}_ft_tofu_${use_lora}_epochs${ft_epochs}_lr${ft_lr}_wd${ft_wd}_${retain_split}_seed42/${retain_checkpoint}"
    unlearn_info="${unlearn_split}_${forget_epochs}_${optim}_${forget_loss}_${forget_lr}_${local_method}_${local_k}_${local_num_retain}_${local_in_scope}_${local_out_scope}_${local_shuffle}"
    python3 aggregate_eval_stat.py \
        retain_result="${retain_path}/eval_results/ds_size300/eval_log_aggregated.json" \
        ckpt_result="${forget_save_dir}/eval_results/ds_size300/eval_log_aggregated.json" \
        method_name=${unlearn_info} \
        save_file="results/${experiment_name}/${model_family}_${ft_lr}_${ft_checkpoint}_unlearn_${unlearn_info}.csv"
done
