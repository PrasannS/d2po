

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2
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

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32
export STEPS=2000

defaults
export BASEMODEL="outputs/models/math/mathsftpartial"


export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

# # # Other commands
export CUDA_VISIBLE_DEVICES=5,6
sh script/dpoplus_script.sh "math" "outputs/data/math/easy2_100k" "functionmath" 29230 "testmathadjust2"