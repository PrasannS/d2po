
accelerate launch --config_file=src/configs/default_single.yaml --main_process_port=${5} \
    src/train_dpo.py \
    --model_name_or_path="$BASEMODEL" --output_dir="outputs/checkpoints/${1}/${2}_${4}_dpo/" \
    --dataset="outputs/data/${1}/${2}" \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --per_device_eval_batch_size=2 \
    --epochs=5 \
    --evaldata="outputs/data/${1}/${3}" \
    --extraevaldata=$EXTRAEVAL \
    --learning_rate=1e-4 \
    --beta=$BETA \
    --save_steps=250 \
    --eval_steps=250 > "outputs/logs/dpo/${1}_${4}_dpo.out"

# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="allenai/tulu-2-7b" --output_dir="dpo/dpoultasmalldistr" \
#     --dataset="data/ultrarmsmall" \
#     --per_device_train_batch_size=1 \
#     --gradient_accumulation_steps=32