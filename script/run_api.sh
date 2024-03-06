contains() {
  case "$1" in
    (*"$2"*) true;;
    (*) false;;
  esac
}

noupdateapi() {

    if contains $3 "facebook"; then 
        echo "using fresh reward"
        REWARD=$3
    else
        REWARD="outputs/models/${1}/${3}"
    fi

    # NOTE that we need to feed things in a specific format    
    python -u src/rmapi.py \
        --model_name=facebook/opt-125m \
        --dataset_name="outputs/data/${1}/${2}" \
        --reward_model_name=$REWARD \
        --save_freq=50 \
        --max_length=256 --batch_size=16 \
        --seed=0 --learning_rate=5e-5 \
        --trainable --goldreward="function${1}" \
        --stopupdates=$SUPDATES \
        --output_dir="outputs/checkpoints/${1}/dynarm_${4}/" \
        --logfile="outputs/results/dynarmlogs/${1}/${3}_${4}.jsonl" \
        --port=${5} \
        --labelthresh=$LABTHRESH \
        --callratio=3 \
        --tracking 
}

# LABTHRESH=0.5
SUPDATES=10000000

export CUDA_VISIBLE_DEVICES=1
# noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "reprodtest" 5000
noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "reprodtest_exact" 5001