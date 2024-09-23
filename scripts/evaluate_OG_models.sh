: '
Evaluate the original model on the perturbed splits.
'
export CUDA_VISIBLE_DEVICES=0

# model_family="qwen2-1.5b"
# retain_path="models/qwen2-1.5b"

model_family="phi"
retain_path="models/phi"

echo "Evaluating ${model_family} model on the perturbed splits."
sleep 10
echo "Done."

# TOFU args #
# if [ $1 == "tofu" ]; then
#     data_name="tofu"
#     data_path="locuslab/TOFU"
#     eval_config_path="eval_tofu.yaml"

#     retain_split="full"
#     FORGET_SPLITS=("forget01_perturbed") # "forget05_perturbed" "forget10_perturbed")

# # KnowUnDo args #
# elif [ $1 == "knowundo" ]; then
#     data_name="knowundo-copyright"
#     data_path="zjunlp/KnowUnDo"
#     eval_config_path="eval_knowundo.yaml"

#     retain_split="full"
#     FORGET_SPLITS=("unlearn")

# else
#     echo "Invalid dataset name."
#     exit 1
# fi

# for forget_split in "${FORGET_SPLITS[@]}"
# do
#     # Evaluate
#     torchrun src/evaluate_util.py --config-name=${eval_config_path} \
#         model_family=${model_family} \
#         model_path=${retain_path} \
#         split=${forget_split} \
#         batch_size=96 \
#         data_name=${data_name}


#     # Aggregate stats
#     og_info="${model_family}_OG_${data_name}_${forget_split}"
#     python3 src/aggregate_eval_stat.py \
#         retain_result="${retain_path}/eval_results/ds_size300/eval_log_aggregated.json" \
#         ckpt_result="${retain_path}/eval_results/ds_size300/eval_log_aggregated.json" \
#         method_name=${og_info} \
#         save_file="results/${model_family}/${data_name}/${og_info}.csv"
# done
