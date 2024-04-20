# need a GPU for local models

BOTTOM=0
TOP=200
MLEN=50
BSIZE=1

# Define a function to run the script with different inputs
run_script() {
    # NOTE that we need to feed things in a specific format
    CKPT_FILE="outputs/checkpoints/${1}/${2}${3}${4}"
    OUTPUT_DIR="outputs/results/genouts/${1}/${2}${4}"
    
    python -u src/eval/generate_outs.py \
        --basemodel="$BASEMODEL" \
        --dset="$DSET" \
        --ckname="$CKPT_FILE" \
        --fname="$OUTPUT_DIR" \
        --bottom=$BOTTOM --top=$TOP  \
        --bsize=$BSIZE \
        --maxlen=$MLEN

    python -u src/evalgold.py  --fname="${OUTPUT_DIR}.jsonl" --gfunct="${1}"
}

DSET="outputs/data/ultra/ultraheld5k"
BASEMODEL="/u/prasanns/research/active-rlhf/outputs/models/ultra/tiny_dpo_tulu"
TOP=200
BSIZE=1
MLEN=256

# export CUDA_VISIBLE_DEVICES=4
# run_script "ultra" "comprm" "/step_" 250

export CUDA_VISIBLE_DEVICES=6

# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 500
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 500
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 300
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 25
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 100
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 200
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 400
# python -u src/evalgold.py  --fname="outputs/results/genouts/eurusrm/test.jsonl" --gfunct="eurusrm"

# run_script "eurusrm" "ppo_eurusrmbaseline" "/step_" 25

# export CUDA_VISIBLE_DEVICES=7
# export CUDA_VISIBLE_DEVICES=4

# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 50
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 75
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 125
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 150
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 175
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 200

# export CUDA_VISIBLE_DEVICES=5
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 225
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 250
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 275
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 325
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 350

# export CUDA_VISIBLE_DEVICES=6
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 450
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 475
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 525
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 550
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 575
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 600


# export CUDA_VISIBLE_DEVICES=7
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 625
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 650
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 675
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 700
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 725
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 750


# export CUDA_VISIBLE_DEVICES=3
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 375
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 400
# run_script "eurusrm" "ppo_mainalgo_32_2" "/step_" 425





# run_script "eurusrm" "ppo_eurusrmbaseline32save2" "/step_" 25
# run_script "eurusrm" "ppo_eurusrmbaseline32save2" "/step_" 50
# run_script "eurusrm" "ppo_eurusrmbaseline32save2" "/step_" 75
# run_script "eurusrm" "ppo_eurusrmbaseline32save2" "/step_" 100
# run_script "eurusrm" "ppo_eurusrmbaseline32save2" "/step_" 125


