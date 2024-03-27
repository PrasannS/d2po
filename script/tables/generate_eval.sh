
BOTTOM=0
TOP=200
MLEN=50
BSIZE=1

# Define a function to run the script with different inputs
run_script() {
    # NOTE that we need to feed things in a specific format
    CKPT_FILE="outputs/checkpoints/${1}/${2}${3}${4}"
    OUTPUT_DIR="outputs/results/genouts/${1}/${2}${4}"
    
    python -u src/eval/generate_outs_old.py \
        --basemodel="$BASEMODEL" \
        --dset="$DSET" \
        --ckname="$CKPT_FILE" \
        --fname="$OUTPUT_DIR" \
        --bottom=$BOTTOM --top=$TOP  \
        --bsize=$BSIZE \
        --maxlen=$MLEN

    python -u src/utils/eval/evalgold.py  --fname="${OUTPUT_DIR}.jsonl" --gfunct="${1}"
}

export CUDA_VISIBLE_DEVICES=0
DSET="outputs/data/ultra/ultraheld5k"
TOP=200
BSIZE=1
# run_script "bagofwords" "dpoplusbow50rm" "/step_" 100

# BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
# run_script "bagofwords" "ppo_justoffpolicy_conf_newalgo_5_5" "/step_" 100
# run_script "bagofwords" "ppo_justoffpolicy_conf_newalgo_5_5" "/step_" 500
# # run_script "bagofwords" "ppo_justoffpolicy_conf_newalgo_goldb8" "/step_" 1000
# # run_script "bagofwords" "ppo_justoffpolicy_conf_newalgo_goldb8" "/step_" 1500

# BASEMODEL="outputs/models/nouns/smalldpo"
# run_script "nouns" "ppo_justoffpolicy_confnoun_newalgo_5_2" "/step_" 100
# run_script "nouns" "ppo_justoffpolicy_confnoun_newalgo_5_2" "/step_" 500
# # run_script "nouns" "ppo_justoffpolicy_confnoun_goldb8" "/step_" 1000
# # run_script "nouns" "ppo_justoffpolicy_confnoun_goldb8" "/step_" 1500

# DSET="outputs/"data/contrastivedistill/wikionpolicyprompts""
# BASEMODEL="outputs/models/contrastivedistill/smalldpo"
# run_script "contrastivedistill" "ppo_justoffpolicy_conf_cdist_10_50_activefix" "/step_" 100
# run_script "contrastivedistill" "ppo_justoffpolicy_conf_cdist_10_50_activefix" "/step_" 500
# run_script "contrastivedistill" "ppo_justoffpolicy_conf_cdist_goldb8" "/step_" 1000
# run_script "contrastivedistill" "ppo_justoffpolicy_conf_cdist_goldb8" "/step_" 1500

BASEMODEL="facebook/opt-125m"

# BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
# run_script "bagofwords" "bow20k_20kdpo_dpo" "/checkpoint-" 1000
# run_script "bagofwords" "bow20k_20kdpo_dpo" "/checkpoint-" 2000
# run_script "bagofwords" "bow20k_20kdpo_dpo" "/checkpoint-" 5000
# run_script "bagofwords" "bow20k_20kdpo_dpo" "/checkpoint-" 10000

# run_script "bagofwords" "ppo_justoffpolicy_conf_newalgo_goldb8" "/step_" 1000
# run_script "bagofwords" "ppo_justoffpolicy_conf_newalgo_goldb8" "/step_" 1500

# BASEMODEL="outputs/models/nouns/smalldpo"
run_script "nouns" "nouns20k_20kdpo_dpo" "/checkpoint-" 1000
run_script "nouns" "nouns20k_20kdpo_dpo" "/checkpoint-" 2000
run_script "nouns" "nouns20k_20kdpo_dpo" "/checkpoint-" 5000
run_script "nouns" "nouns20k_20kdpo_dpo" "/checkpoint-" 10000
# run_script "nouns" "ppo_justoffpolicy_confnoun_goldb8" "/step_" 1000
# run_script "nouns" "ppo_justoffpolicy_confnoun_goldb8" "/step_" 1500

DSET="outputs/data/contrastivedistill/wikionpolicyprompts"
# BASEMODEL="outputs/models/contrastivedistill/smalldpo"
run_script "contrastivedistill" "opt20k_20kdpo_dpo" "/checkpoint-" 1000
run_script "contrastivedistill" "opt20k_20kdpo_dpo" "/checkpoint-" 2000
run_script "contrastivedistill" "opt20k_20kdpo_dpo" "/checkpoint-" 5000
run_script "contrastivedistill" "opt20k_20kdpo_dpo" "/checkpoint-" 10000
