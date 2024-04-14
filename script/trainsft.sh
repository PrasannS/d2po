
sftrun() {
    # NOTE that we need to feed things in a specific format    

    python -u src/trainsft.py \
        --dataset_name="outputs/data/${1}/${2}" \
        --output_dir="checkpoints/${1}/${2}_sft_${4}" \
        --save_steps=4000 \
        --num_train_epochs=1 \
        --learning_rate=1e-5 \
        --model_name=${3} \
        --per_device_train_batch_size=16 \
        --per_device_eval_batch_size=16 \
        --warmup_steps=50 \
        --logging_steps=10
}

# export CUDA_VISIBLE_DEVICES=2
# sftrun "math" 'easy2_100k' "facebook/opt-125m"

export CUDA_VISIBLE_DEVICES=2
sftrun "math" 'mathsfthuge' "facebook/opt-1.3b" "big"
# sftrun "math" 'mathsfthuge' "facebook/opt-125m"