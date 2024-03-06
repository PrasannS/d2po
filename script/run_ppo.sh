contains() {
  case "$1" in
    (*"$2"*) true;;
    (*) false;;
  esac
}

STEPS=2000
CFG=src/configs/ppo_2gpu.yaml
dpoplus_script() {
    # NOTE that we need to feed things in a specific format    

    if contains $3 "http"; then 
        echo "using http reward"
        REWARD=$3
    else
        REWARD="outputs/models/rewards/${1}/${3}_rm"
    fi


    if contains $5 "normppo"; then 
        echo "using normal PPO objective"
        KLP="kl"
        OSAMP=4
    else
        echo "using DPO plus objective"
        KLP="dpoplus"
        OSAMP=2
    fi
    
    echo "config file $CFG"
    accelerate launch --multi_gpu --config_file=$CFG --main_process_port=${4} \
        --num_machines 1  --num_processes 2 \
        src/train_ppo.py --log_with=wandb \
        --model_name=$BASEMODEL \
        --dataset_name="${2}" \
        --reward_model_name=$REWARD \
        --adafactor=False \
        --save_freq=25 \
        --max_length=$MLEN --batch_size=32 \
        --mini_batch_size=32 \
        --gradient_accumulation_steps=1 \
        --ppo_epochs=1 --seed=$SEED --learning_rate=5e-5 \
        --early_stopping=False --output_dir=outputs/checkpoints/${1}/ppo_${5} \
        --init_kl_coef=0.05 --steps=$STEPS \
        --oversample=$OSAMP \
        --temperature=1 \
        --rollout_strategy=normal \
        --gen_bsize=32 \
        --kl_penalty="$KLP" --keep_long=$KEEPLONG \
        --save_rollouts=True
    
    # TODO undo rollout saving whenever we want to do that
}

SEED=0
KEEPLONG=0
MLEN=50
BASEMODEL="facebook/opt-125m"
export CUDA_VISIBLE_DEVICES=2,3
dpoplus_script "bagofwords" "ultra" "http://127.0.0.1:5001/train" 29516 "activereprod_exact"

# export CUDA_VISIBLE_DEVICES=3,4
# dpoplus_script "bagofwords" "ultra/ultrafeeddiff" "125rerun" 29517 "sanityrm125"

# export CUDA_VISIBLE_DEVICES=5,6
# dpoplus_script "bagofwords" "ultra/ultrafeeddiff" "functionbagofwords" 29516 "sanitygoldnofa"