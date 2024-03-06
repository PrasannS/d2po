# SH File for running the whole shebang, multiple jobs

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export LABTHRESH=0.3
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0
export MLEN=50
export BASEMODEL="facebook/opt-125m"

export REDOBATCH=1
export LABRATIO=0.25
export ATYPE="rand"

# RAND SUB_SAMP + REPLAY (5 batch)

export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "expbow50" "rand_subsamp_replay5" 5002 & 
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5002/train" 29517 "rand_subsamp_replay5"
jobs
pkill -f "rand_subsamp"
jobs


# RAND SUB_SAMP + REPLAY (15 batch)

export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "expbow50" "rand_subsamp_replay15" 5002 & 
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5002/train" 29517 "rand_subsamp_replay15"
jobs
pkill -f "rand_subsamp"
jobs

# RAND SUB_SAMP + IN-DOMAIN PROMPT DATA (maybe with expbow)

export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "expbow50" "rand_subsamp_indomain" 5002 & 
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5002/train" 29517 "rand_subsamp_indomain"
jobs
pkill -f "rand_subsamp"
jobs

# BEST TECHNIQUE (ALL IN ONE)

export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "expbow50" "conf_all" 5002 & 
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5002/train" 29517 "conf_all"
jobs
pkill -f "conf_all"
jobs