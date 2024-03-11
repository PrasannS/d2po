# SH File for running the whole shebang, multiple jobs
contains() {
  case "$1" in
    (*"$2"*) true;;
    (*) false;;
  esac
}

if contains $3 "facebook"; then 
    echo "using fresh reward"
    REWARD=$3
else
    REWARD="outputs/models/${1}/${3}"
fi

# NOTE that we added a necessary GREWARD env variable
# NOTE that we need to feed things in a specific format    
python -u src/rmapi.py \
    --model_name=facebook/opt-125m \
    --dataset_name="outputs/data/${1}/${2}" \
    --reward_model_name=$REWARD \
    --save_freq=50 \
    --max_length=512 --batch_size=$APBSIZE \
    --seed=0 --learning_rate=$ULR \
    --trainable --goldreward="function$GREWARD" \
    --stopupdates=$SUPDATES \
    --output_dir="outputs/checkpoints/${1}/dynarm_${4}/" \
    --logfile="outputs/results/dynarmlogs/${1}/${3}_${4}.jsonl" \
    --port=${5} \
    --labelthresh=$LABTHRESH \
    --callratio=$CRATIO \
    --redo_batches=$REDOBATCH \
    --relab_criteria=$ATYPE \
    --relabel_ratio=$LABRATIO \
    --tracking > "outputs/logs/api/${1}_${4}.out"