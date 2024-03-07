# TEMPOARY FILE TO AVOID MESSING UP RUNNING JOBS

contains() {
  case "$1" in
    (*"$2"*) true;;
    (*) false;;
  esac
}

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
    --max_length=$MLEN --batch_size=8 \
    --mini_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --ppo_epochs=1 --seed=$SEED --learning_rate=5e-5 \
    --early_stopping=False --output_dir=outputs/checkpoints/${1}/ppo_${5} \
    --init_kl_coef=0.05 --steps=$STEPS \
    --oversample=$OSAMP \
    --temperature=1 \
    --rollout_strategy=normal \
    --gen_bsize=32 \
    --kl_penalty="$KLP" --keep_long=$KEEPLONG \
    --save_rollouts=True > "outputs/logs/ppo/${1}_${5}.out"