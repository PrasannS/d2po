# OFF POLICY VERSIONS OF ALL THE ACTIVE THINGS: 

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export SUPDATES=10000000
export SEED=0
export KEEPLONG=10

# CONTRASTIVE DISTILL
defaults() {
    export MLEN=50
    export ATYPE="conf"
    export ULR=1e-4
    export CRATIO=1
}

export USEDPO=0
export ONLYOLDUPDATES=0

# # BEST TECHNIQUE (ALL IN ONE)
defaults
export DPOBATCHSIZE=32
export MBSIZE=8
export GBSIZE=32


export CUDA_VISIBLE_DEVICES=0,1

export BASEMODEL="outputs/models/nouns/smalldpo"
# 4 updates on gold whenever we get it
export PPOUPDATES=1

sh script/dpoplus_script.sh "nouns" "ultra" "tiny" 29523 "noun_initrmbaseline"


export CUDA_VISIBLE_DEVICES=0,1

export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
# 4 updates on gold whenever we get it
export PPOUPDATES=4

sh script/dpoplus_script.sh "bagofwords" "ultra" "bowtiny" 29524 "bow_initrmbaseline"


export CUDA_VISIBLE_DEVICES=0,1

export BASEMODEL="outputs/models/contrastivedistill/smalldpo"
# 4 updates on gold whenever we get it
export PPOUPDATES=4
sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "tiny" 29525 "cdist_initrmbaseline"
