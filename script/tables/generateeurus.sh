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


DSET="outputs/data/math/mathppoinps200k"
BASEMODEL="/u/prasanns/research/active-rlhf/outputs/models/math/mathbigdata1b"
# DSET="outputs/data/paraphrase/parappoinps"
# BASEMODEL="facebook/opt-125m"
TOP=200
BSIZE=1
MLEN=75
export CUDA_VISIBLE_DEVICES=2
# run_script "math" "ppo_1bbig_128_8" "/step_" 25
# # run_script "math" "ppo_mathattempt2bigf2" "orig" ""
# run_script "math" "ppo_1bbig_128_8" "/step_" 50
# run_script "math" "ppo_1bbig_128_8" "/step_" 75
# run_script "math" "ppo_1bbig_128_8" "/step_" 100

# run_script "math" "ppo_1bbig_128_8" "/step_" 125
# # run_script "math" "ppo_mathattempt2bigf2" "orig" ""
# run_script "math" "ppo_1bbig_128_8" "/step_" 150
# run_script "math" "ppo_1bbig_128_8" "/step_" 175
# run_script "math" "ppo_1bbig_128_8" "/step_" 200

# export CUDA_VISIBLE_DEVICES=2
# run_script "math" "ppo_mathattempt2bigf2" "/step_" 125
# run_script "math" "ppo_mathattempt2bigf2" "/step_" 150
# run_script "math" "ppo_mathattempt2bigf2" "/step_" 175
# run_script "math" "ppo_mathattempt2bigf2" "/step_" 200
# run_script "math" "ppo_mathattempt2bigf2" "/step_" 225
# run_script "math" "ppo_mathattempt2bigf2" "/step_" 250
# export CUDA_VISIBLE_DEVICES=1
# # run_script "math" "ppo_1bbigfreq" "/step_" 100
# run_script "math" "ppo_1bbigfreq" "/step_" 25
# run_script "math" "ppo_1bbigfreq" "/step_" 50
# run_script "math" "ppo_1bbigfreq" "/step_" 75

# export CUDA_VISIBLE_DEVICES=4
# # run_script "math" "ppo_1bbigfreq" "/step_" 100
# run_script "math" "ppo_1bbigfreq" "/step_" 125
# run_script "math" "ppo_1bbigfreq" "/step_" 150
# run_script "math" "ppo_1bbigfreq" "/step_" 175
# # run_script "math" "ppo_1bbigfreq" "/step_" 100

# export CUDA_VISIBLE_DEVICES=5
# # run_script "math" "ppo_1bbigfreq" "/step_" 100
# run_script "math" "ppo_1bbigfreq" "/step_" 225
# run_script "math" "ppo_1bbigfreq" "/step_" 250
# run_script "math" "ppo_1bbigfreq" "/step_" 275

# export CUDA_VISIBLE_DEVICES=6
# # run_script "math" "ppo_1bbigfreq" "/step_" 100
# run_script "math" "ppo_1bbigfreq" "/step_" 325
# run_script "math" "ppo_1bbigfreq" "/step_" 350
# run_script "math" "ppo_1bbigfreq" "/step_" 375

export CUDA_VISIBLE_DEVICES=7
# run_script "math" "ppo_1bbigfreq" "/step_" 100
# run_script "math" "ppo_1bbigfreq" "/step_" 425
# run_script "math" "ppo_1bbigfreq" "/step_" 450
# run_script "math" "ppo_1bbigfreq" "/step_" 475
# run_script "math" "ppo_1bbigfreq" "/step_" 200
# run_script "math" "ppo_1bbigfreq" "/step_" 300
# run_script "math" "ppo_1bbigfreq" "/step_" 400
# run_script "math" "ppo_1bbigfreq" "/step_" 500
# export CUDA_VISIBLE_DEVICES=5

# run_script "math" "ppo_1bbigfreq" "/step_" 600
# run_script "math" "ppo_1bbigfreq" "/step_" 700
# run_script "math" "ppo_1bbigfreq" "/step_" 800
# run_script "math" "ppo_1bbigfreq" "/step_" 900
# run_script "math" "ppo_1bbigfreq" "/step_" 1000
# export CUDA_VISIBLE_DEVICES=6

# run_script "math" "ppo_1bbigfreq" "/step_" 1100

# run_script "math" "ppo_1bbigfreq" "/step_" 1200
# run_script "math" "ppo_1bbigfreq" "/step_" 1300
# run_script "math" "ppo_1bbigfreq" "/step_" 1400
# run_script "math" "ppo_1bbigfreq" "/step_" 1500
export CUDA_VISIBLE_DEVICES=7

# run_script "math" "ppo_1bbigfreq" "/step_" 1600
# run_script "math" "ppo_1bbigfreq" "/step_" 1700
# run_script "math" "ppo_1bbigfreq" "/step_" 1800
# run_script "math" "ppo_1bbigfreq" "/step_" 1900
# run_script "math" "ppo_1bbigfreq" "/step_" 2000

# export CUDA_VISIBLE_DEVICES=4
# # run_script "ultra" "comprm" "/step_" 250

# run_script "eurusrm" "offpolicydata1k_euroffv1_dpo" "/checkpoint-" 25
# run_script "eurusrm" "offpolicydata1k_euroffv1_dpo" "/checkpoint-" 50
# run_script "eurusrm" "offpolicydata1k_euroffv1_dpo" "/checkpoint-" 75
# run_script "eurusrm" "offpolicydata1k_euroffv1_dpo" "/checkpoint-" 100

# export CUDA_VISIBLE_DEVICES=5
# run_script "eurusrm" "offpolicydata1k_euroffv1_dpo" "/checkpoint-" 300
# export CUDA_VISIBLE_DEVICES=6
# run_script "eurusrm" "offpolicydata1k_euroffv1_dpo" "/checkpoint-" 400
# export CUDA_VISIBLE_DEVICES=7
# run_script "eurusrm" "offpolicydata1k_euroffv1_dpo" "/checkpoint-" 500
# export CUDA_VISIBLE_DEVICES=7
# run_script "eurusrm" "offpolicydata1k_euroffv1_dpo" "/checkpoint-" 600


# export CUDA_VISIBLE_DEVICES=6
# run_script "eurusrm" "ppo_mainalgo_32_4justoffpolicy" "/step_" 25
# run_script "eurusrm" "ppo_mainalgo_32_4justoffpolicy" "/step_" 50
# run_script "eurusrm" "ppo_mainalgo_32_4justoffpolicy" "/step_" 75
# run_script "eurusrm" "ppo_mainalgo_32_4justoffpolicy" "/step_" 100

# export CUDA_VISIBLE_DEVICES=7
# run_script "eurusrm" "ppo_mainalgo_32_4justoffpolicy" "/step_" 125
# run_script "eurusrm" "ppo_mainalgo_32_4justoffpolicy" "/step_" 150
# run_script "eurusrm" "ppo_mainalgo_32_4justoffpolicy" "/step_" 175
# run_script "eurusrm" "ppo_mainalgo_32_4justoffpolicy" "/step_" 200

# export CUDA_VISIBLE_DEVICES=3
# run_script "eurusrm" "ppo_mainalgo_32_4justoffpolicy" "/step_" 225
# run_script "eurusrm" "ppo_mainalgo_32_4justoffpolicy" "/step_" 250
# run_script "eurusrm" "ppo_mainalgo_32_4justoffpolicy" "/step_" 275
# run_script "eurusrm" "ppo_mainalgo_32_4justoffpolicy" "/step_" 300


# export CUDA_VISIBLE_DEVICES=0

# # run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/orig" ""
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 100
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 200
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 300
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 400
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 500
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 600
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 700
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 800
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 900

# export CUDA_VISIBLE_DEVICES=1
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 1000
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 1100
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 1200
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 1300
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 1400
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 1500
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 1600
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 1700
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 1800
# run_script "paraphrase" "ppo_parconf_newalgo_256_16_ndata" "/step_" 1900

export CUDA_VISIBLE_DEVICES=2
# run_script "paraphrase" "ppo_opogold32paranew" "/step_" 25
# run_script "paraphrase" "ppo_opogold32paranew" "/step_" 50
# run_script "paraphrase" "ppo_opogold32paranew" "/step_" 75
# run_script "paraphrase" "ppo_opogold32paranew" "/step_" 100
# run_script "paraphrase" "ppo_opogold32paranew" "/step_" 125
# run_script "paraphrase" "ppo_opogold32paranew" "/step_" 150
# run_script "paraphrase" "ppo_opogold32paranew" "/step_" 175
# run_script "paraphrase" "ppo_opogold32paranew" "/step_" 200
# run_script "paraphrase" "ppo_opogold32paranew" "/step_" 225
# run_script "paraphrase" "ppo_opogold32paranew" "/step_" 250





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


