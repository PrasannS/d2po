# OFF POLICY VERSIONS OF ALL THE ACTIVE THINGS: 
# IMPORTANT BASELINE, WHAT HAPPENS WITH MORE DPO UPDATES ON THE SAME THING
export CFG=src/configs/ppo_2gpu.yaml
export STEPS=251
export SUPDATES=10000000
export SEED=0
export KEEPLONG=10

# CONTRASTIVE DISTILL
defaults() {
    export BASEMODEL="outputs/models/contrastivedistill/smalldpo"
    export MLEN=50
    export ATYPE="conf"
    export ULR=1e-4
    export CRATIO=1
}

export USEDPO=0
export ONLYOLDUPDATES=0
export PPOUPDATES=4
defaults

# # BEST TECHNIQUE (ALL IN ONE)
# defaults
# export ATYPE="conf"
# export UEPOCHS=3
# export APBSIZE=16
# export GREWARD="contrastivedistill"

export DPOBATCHSIZE=8
export MBSIZE=8
export GBSIZE=8

# # Other commands
# export CUDA_VISIBLE_DEVICES=0,1
# sh script/ipoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "functioncontrastivedistill" 29523 "contdistb8_moreupdates_4ups_ipo"
# jobs
# # pkill -f "justoffpolicy_conf_cdist_100_50_activefix"
# jobs

# # NOUNS ___________
# defaults
# export BASEMODEL="outputs/models/nouns/smalldpo"

# export DPOBATCHSIZE=8
# export MBSIZE=8
# export GBSIZE=8

# # Other commands
# export CUDA_VISIBLE_DEVICES=2,3
# sh script/ipoplus_script.sh "nouns" "ultra" "functionnouns" 29524 "confnoun_goldb8_4ups_ipo"


# # #  __________________--
defaults
export BASEMODEL="facebook/opt-125m"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

# # # Other commands
# export CUDA_VISIBLE_DEVICES=6,7
# sh script/dpoplus_script.sh "paraphrase" "outputs/data/paraphrase/parappoinps" "functionparaphrase" 29526 "opogold32paranew"

export SFREQ=12
export SEED=4
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "unique_nns" "ultra" "functionunique_nns" 29527 "conf_newalgo_goldb8_4ups_dpos4"
export SEED=5
sh script/dpoplus_script.sh "unique_nns" "ultra" "functionunique_nns" 29527 "conf_newalgo_goldb8_4ups_dpos5"

# export SFREQ=12
# export DPOBATCHSIZE=8
# export BASEMODEL="outputs/models/math/mathbigdata1b"
# export MBSIZE=8
# export GBSIZE=8
# export PPOUPDATES=2
# export MLEN=75
# export CUDA_VISIBLE_DEVICES=6,7
# export SEED=3
# sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps200k" "functionmath" 29526 "opobase8mathseed3"
# export SEED=4
# sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps200k" "functionmath" 29526 "opobase8mathseed4"