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
    REWARD="outputs/models/${1}/${3}_rm"
fi

if contains $5 "normppo"; then 
    echo "using normal PPO objective"
    KLP="kl"
    OSAMP=2
else
    echo "using DPO plus objective"
    KLP="dpoplus"
    OSAMP=2
fi

if contains $5 "justoffpolicy"; then 
    echo "using only off policy rollouts"
    GENJSON="src/configs/offpolicygen.json"
else
    echo "using normal rollouts"
    GENJSON="none"
fi


echo $SFREQ
echo "config file $CFG"
accelerate launch --multi_gpu --config_file=$CFG --main_process_port=${4} \
    --num_machines 1  --num_processes 2 \
    src/train_ppo.py --log_with=wandb \
    --model_name=$BASEMODEL \
    --dataset_name="${2}" \
    --reward_model_name=$REWARD \
    --adafactor=False \
    --save_freq=$SFREQ \
    --max_length=$MLEN --batch_size=$DPOBATCHSIZE \
    --mini_batch_size=$MBSIZE \
    --gradient_accumulation_steps=1 \
    --ppo_epochs=$PPOUPDATES --seed=$SEED --learning_rate=5e-5 \
    --early_stopping=False --output_dir=outputs/checkpoints/${1}/ppo_${5} \
    --init_kl_coef=0.05 --steps=$STEPS \
    --oversample=$OSAMP \
    --temperature=1 \
    --rollout_strategy=normal \
    --gen_bsize=$GBSIZE \
    --kl_penalty="$KLP" --keep_long=$KEEPLONG \
    --generators_json=$GENJSON \
    --save_rollouts=True > "outputs/logs/ppo/${1}_${5}.out" 