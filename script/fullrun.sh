# SH File for running the whole shebang, multiple jobs

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=5
export LABTHRESH=0.3
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0
export MLEN=50
export BASEMODEL="facebook/opt-125m"

export CUDA_VISIBLE_DEVICES=2
# noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "expbow50" "sanity" 5001 & 
PID=$!
echo $PID
# Other commands
export CUDA_VISIBLE_DEVICES=3,4
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5001/train" 29516 "sanity"
jobs
killall "5001"
jobs