
sftrun() {
    # NOTE that we need to feed things in a specific format    

    accelerate launch --multi_gpu --num_processes=2 src/trainsft.py \
        --dataset_name="outputs/data/${1}/${2}" \
        --output_dir="checkpoints/${1}/${2}_sft_${4}" \
        --save_steps=4000 \
        --num_train_epochs=1 \
        --learning_rate=5e-5 \
        --model_name=${3} \
        --per_device_train_batch_size=$BATCHSIZE \
        --per_device_eval_batch_size=$BATCHSIZE \
        --warmup_steps=10 \
        --logging_steps=10
}

# export CUDA_VISIBLE_DEVICES=2
# sftrun "math" 'easy2_100k' "facebook/opt-125m"
# BATCHSIZE=16
BATCHSIZE=2
export CUDA_VISIBLE_DEVICES=6,7
# sftrun "math" 'mbestofnsft' "outputs/models/math/randbigsft" "bonsft"
# sftrun "math" 'sftcompsmall' "outputs/models/math/randbigsft" "goldsft"
sftrun "easymusr" 'easymusrsft' "allenai/tulu-2-7b" "goldsft"

# sftrun "math" 'mathsfthuge' "facebook/opt-125m"