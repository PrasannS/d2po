# python src/merge_adapter.py \
#     --adapter_model_name="/u/prasanns/research/active-rlhf/checkpoints/math/mathsfthuge_sft/checkpoint-50000" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="/u/prasanns/research/active-rlhf/outputs/models/math/mathbigdata125"

# python src/merge_adapter.py \
#     --adapter_model_name="/u/prasanns/research/active-rlhf/outputs/checkpoints/math/dynarm_1bjustoffpolicyrmupper/step_3000" \
#     --base_model_name="outputs/models/math/mathbigdata1b" \
#     --output_name="outputs/models/math/math50krm"

# python src/merge_adapter.py \
#     --adapter_model_name="/u/prasanns/research/active-rlhf/outputs/checkpoints/math/mathwarm_mathinitstart_rm/checkpoint-240" \
#     --base_model_name="outputs/models/math/mathbigdata1b" \
#     --output_name="outputs/models/math/mathwarm1b"

# python src/merge_adapter.py \
#     --adapter_model_name="/u/prasanns/research/active-rlhf/outputs/checkpoints/math/offp40knotie_mathbigoffp_rm/checkpoint-6700" \
#     --base_model_name="outputs/models/math/mathbigdata1b" \
#     --output_name="outputs/models/math/mathbig40koff_rm"

python src/merge_adapter.py \
    --adapter_model_name="/u/prasanns/research/active-rlhf/outputs/checkpoints/unique_nns/fullnpref_bigrmnew40k_rm/checkpoint-6800" \
    --base_model_name="facebook/opt-125m" \
    --output_name="outputs/models/unique_nns/newbigrm"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/bagofwords/3kprefs_smalldpo_dpo/checkpoint-500" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="../active-rlhf/outputs/models/bagofwords/bowtiny_dpo"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/bagofwords/nozero100k_1bnofa_rm/checkpoint-9000" \
#     --base_model_name="facebook/opt-1.3b" \
#     --output_name="models/rewards/bagofwords/1bnofarm"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/80rmv2/_peft_last_checkpoint" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/rewards/expbow80"

# TODO look into trying out earlier checkpoints as well
# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/30rmv2/_peft_last_checkpoint" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/rewards/expbow30"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/reversebow/50krmnp/best_checkpoint" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/rewards/revbow/revrm50k"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/reversebow/truncrmnp/best_checkpoint" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/rewards/revbow/revrmtrunc"

# export CUDA_VISIBLE_DEVICES=7
# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="checkpoints/bagofwords/dpoplusbow50rm/step_100" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/bagofwords/50rmppo_s100_sft"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="checkpoints/bagofwords/dpoplusbow50rm/step_200" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/bagofwords/50rmppo_s200_sft"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/bagofwords/bowreversedata_reversebow_dpo/checkpoint-7000" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/bagofwords/tinybow_sft"

# export CUDA_VISIBLE_DEVICES=7
# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="checkpoints/13rmultratrunc/checkpoint-10500" \
#     --base_model_name="allenai/tulu-2-13b" \
#     --output_name="models/rewards/13btruncrm"

# export CUDA_VISIBLE_DEVICES=7
# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/einsteinsftllama7b/checkpoint-10000" \
#     --base_model_name="/u/prasanns/research/rlhf-length-biases/models/llama" \
#     --output_name="models/sft7beinstein"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/rmself25/checkpoint-7000" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/rewards/bowrmself25"



