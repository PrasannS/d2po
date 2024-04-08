
BOTTOM=0
TOP=50
MLEN=50
BSIZE=1

# Define a function to run the script with different inputs
run_script() {
    # NOTE that we need to feed things in a specific format
    CKPT_FILE="outputs/checkpoints/${1}/${2}${3}${4}"
    OUTPUT_DIR="outputs/results/genouts/${1}/${2}${4}"
    echo $CUDA_VISIBLE_DEVICES

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

justgen() {
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

    #python -u src/utils/eval/evalgold.py  --fname="${OUTPUT_DIR}.jsonl" --gfunct="${1}"
}


export CUDA_VISIBLE_DEVICES=3

DSET="outputs/data/ultra/ultraheld5k"
TOP=100
BSIZE=1
BASEMODEL="facebook/opt-125m"
export CUDA_VISIBLE_DEVICES=1

# run_script "bagofwords" "bowsynth50knozeros_ipo100_ipo" "/checkpoint-" 250
# run_script "bagofwords" "bowsynth50knozeros_ipo100_ipo" "/checkpoint-" 500
# run_script "bagofwords" "bowsynth50knozeros_ipo100_ipo" "/checkpoint-" 1000
# run_script "bagofwords" "bowsynth50knozeros_ipo100_ipo" "/checkpoint-" 5000
# run_script "bagofwords" "bowsynth50knozeros_ipo100_ipo" "/checkpoint-" 15000
# run_script "bagofwords" "bowsynth50knozeros_ipo100_ipo" "/checkpoint-" 30000

# export CUDA_VISIBLE_DEVICES=2
# run_script "bagofwords" "bowsynth50knozeros_ipo1_ipo" "/checkpoint-" 250
# run_script "bagofwords" "bowsynth50knozeros_ipo1_ipo" "/checkpoint-" 500
# run_script "bagofwords" "bowsynth50knozeros_ipo1_ipo" "/checkpoint-" 1000
# run_script "bagofwords" "bowsynth50knozeros_ipo1_ipo" "/checkpoint-" 5000
# run_script "bagofwords" "bowsynth50knozeros_ipo1_ipo" "/checkpoint-" 15000
# run_script "bagofwords" "bowsynth50knozeros_ipo1_ipo" "/checkpoint-" 30000
# export CUDA_VISIBLE_DEVICES=3

# run_script "bagofwords" "bowsynth50knozeros_ipo10_ipo" "/checkpoint-" 250
# run_script "bagofwords" "bowsynth50knozeros_ipo10_ipo" "/checkpoint-" 500
# run_script "bagofwords" "bowsynth50knozeros_ipo10_ipo" "/checkpoint-" 1000
# run_script "bagofwords" "bowsynth50knozeros_ipo10_ipo" "/checkpoint-" 5000
# run_script "bagofwords" "bowsynth50knozeros_ipo10_ipo" "/checkpoint-" 15000
# run_script "bagofwords" "bowsynth50knozeros_ipo10_ipo" "/checkpoint-" 30000
# export CUDA_VISIBLE_DEVICES=4

# run_script "bagofwords" "bowsynth50knozeros_ipopt1_ipo" "/checkpoint-" 250
# run_script "bagofwords" "bowsynth50knozeros_ipopt1_ipo" "/checkpoint-" 500
# run_script "bagofwords" "bowsynth50knozeros_ipopt1_ipo" "/checkpoint-" 1000
# run_script "bagofwords" "bowsynth50knozeros_ipopt1_ipo" "/checkpoint-" 5000
# run_script "bagofwords" "bowsynth50knozeros_ipopt1_ipo" "/checkpoint-" 15000
# run_script "bagofwords" "bowsynth50knozeros_ipopt1_ipo" "/checkpoint-" 30000
export CUDA_VISIBLE_DEVICES=5

run_script "bagofwords" "bowsynth50knozeros_ipopt001_ipo" "/checkpoint-" 250
run_script "bagofwords" "bowsynth50knozeros_ipopt001_ipo" "/checkpoint-" 500
run_script "bagofwords" "bowsynth50knozeros_ipopt001_ipo" "/checkpoint-" 1000
run_script "bagofwords" "bowsynth50knozeros_ipopt001_ipo" "/checkpoint-" 5000
run_script "bagofwords" "bowsynth50knozeros_ipopt001_ipo" "/checkpoint-" 15000
run_script "bagofwords" "bowsynth50knozeros_ipopt001_ipo" "/checkpoint-" 30000


# BASEMODEL="outputs/models/ultra/tiny_dpo_tulu"
# for i in $(seq 25 25 225)
# do
#   run_script "ultra" "ppo_conf_active_newalgoultra" "/step_" "$i"
# done


# for i in $(seq 25 25 125)
# do
#   run_script "ultra" "fullgolddpo" "/step_" "$i"
# done


# run_script "bagofwords" "dpoplusbow50rm" "/step_" 100

BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"

# for i in $(seq 100 100 2000)
# do
#   run_script "bagofwords" "ppo_conf_newalgo_5testnewseed2_dpoapi" "/step_" "$i"
# done

# for i in $(seq 25 25 500)
# do
#   run_script "bagofwords" "ppo_confbow_goldb8_selfreward" "/step_" "$i"
# done


# # run_script "bagofwords" "ppo_conf_newalgo_5testnewseed3" "/step_" 2000


# BASEMODEL="outputs/models/nouns/smalldpo"

# for i in $(seq 25 25 250)
# do
#   run_script "nouns" "ppo_confnoun_goldb8_selfreward" "/step_" "$i"
# done
# for i in $(seq 600 100 2000)
# do
#   run_script "nouns" "ppo_confnoun_newalgo_2_5_seed2" "/step_" "$i"
# done

# for i in $(seq 25 25 250)
# do
#   run_script "nouns" "ppo_confnoun_goldb8_4ups" "/step_" "$i"
# done

# run_script "bagofwords" "ppo_conf_newalgo_5testnewseed3" "/step_" "$j"
# run_script "bagofwords" "ppo_justoffpolicy_conf_newalgo_5_5" "/step_" 100
# run_script "bagofwords" "ppo_justoffpolicy_conf_newalgo_5_5" "/step_" 500
# # run_script "bagofwords" "ppo_justoffpolicy_conf_newalgo_goldb8" "/step_" 1000
# # run_script "bagofwords" "ppo_justoffpolicy_conf_newalgo_goldb8" "/step_" 1500

# BASEMODEL="outputs/models/nouns/smalldpo"
# run_script "nouns" "ppo_justoffpolicy_confnoun_newalgo_5_2" "/step_" 100
# run_script "nouns" "ppo_justoffpolicy_confnoun_newalgo_5_2" "/step_" 500
# # run_script "nouns" "ppo_justoffpolicy_confnoun_goldb8" "/step_" 1000
# # run_script "nouns" "ppo_justoffpolicy_confnoun_goldb8" "/step_" 1500

DSET="outputs/"data/contrastivedistill/wikionpolicyprompts""
BASEMODEL="outputs/models/contrastivedistill/smalldpo"

# run_script "contrastivedistill" "ppo_justoffpolicy_conf_cdist_10_50_activefix" "/step_" 100
# run_script "contrastivedistill" "ppo_justoffpolicy_conf_cdist_10_50_activefix" "/step_" 500
# run_script "contrastivedistill" "ppo_justoffpolicy_conf_cdist_goldb8" "/step_" 1000
# run_script "contrastivedistill" "ppo_justoffpolicy_conf_cdist_goldb8" "/step_" 1500

# for i in $(seq 25 25 250)
# do
#   run_script "contrastivedistill" "ppo_confcdist_goldb8_selfreward" "/step_" "$i"
# done

# for i in $(seq 500 100 2000)
# do
#   run_script "contrastivedistill" "ppo_rand_cdist_10_5_activefixseed2api" "/step_" "$i"
# done

# for i in $(seq 25 25 250)
# do
#   run_script "contrastivedistill" "ppo_contdistb8_moreupdates_4ups" "/step_" "$i"
# done

BASEMODEL="facebook/opt-125m"

# BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
# run_script "bagofwords" "bow20k_20kdpo_dpo" "/checkpoint-" 1000
# run_script "bagofwords" "bow20k_20kdpo_dpo" "/checkpoint-" 2000
# run_script "bagofwords" "bow20k_20kdpo_dpo" "/checkpoint-" 5000
# run_script "bagofwords" "bow20k_20kdpo_dpo" "/checkpoint-" 10000

# run_script "bagofwords" "ppo_justoffpolicy_conf_newalgo_goldb8" "/step_" 1000
# run_script "bagofwords" "ppo_justoffpolicy_conf_newalgo_goldb8" "/step_" 1500

# BASEMODEL="outputs/models/nouns/smalldpo"
# run_script "nouns" "nouns20k_20kdpo_dpo" "/checkpoint-" 1000
# run_script "nouns" "nouns20k_20kdpo_dpo" "/checkpoint-" 2000
# run_script "nouns" "nouns20k_20kdpo_dpo" "/checkpoint-" 5000
# run_script "nouns" "nouns20k_20kdpo_dpo" "/checkpoint-" 10000
# run_script "nouns" "ppo_justoffpolicy_confnoun_goldb8" "/step_" 1000
# run_script "nouns" "ppo_justoffpolicy_confnoun_goldb8" "/step_" 1500

export CUDA_VISIBLE_DEVICES=6

DSET="outputs/data/contrastivedistill/wikionpolicyprompts"
# BASEMODEL="outputs/models/contrastivedistill/smalldpo"
# BASEMODEL="facebook/opt-125m"
# run_script "contrastivedistill" "opt4k_4kdpo_dpo" "/checkpoint-" 250
# run_script "contrastivedistill" "opt4k_4kdpo_dpo" "/checkpoint-" 500
# run_script "contrastivedistill" "opt4k_4kdpo_dpo" "/checkpoint-" 1000
# run_script "contrastivedistill" "opt4k_4kdpo_dpo" "/checkpoint-" 10000
