export CUDA_VISIBLE_DEVICES=4
# python -u notebooks/testmath.py "outputs/checkpoints/math/mdpomargin2_bondpomathambig_dpo/final_checkpoint" "sampdpo"

python -u notebooks/testmath.py "outputs/checkpoints/math/mdpomargingver2_golddpomathambig_dpo" "gsdpo"
