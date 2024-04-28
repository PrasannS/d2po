from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PPOArguments:
    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="facebook/opt-125m", metadata={"help": "the model name"})
    reward_model_name: Optional[str] = field(default="function:bagofwords", metadata={"help": "the reward model name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(default="ultra", metadata={"help": "the dataset name"})
    # reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    kl_penalty: Optional[str] = field(default="kl", metadata={"help": "kl penalty setup, can use dpoplus for that"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    max_length: Optional[int] = field(default=50, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    self_reward_steps: Optional[int] = field(default=0, metadata={"help": "how many self reward prefs to get when we get them"})
    self_reward_rollouts: Optional[int] = field(default=32*5, metadata={"help": "how many self reward prefs to get when we get them"})
    ppo_epochs: Optional[int] = field(default=1, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    save_rollouts: Optional[bool] = field(default=False, metadata={"help": "save rollouts, rewards to file"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
       default=0,
       metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    omit_long: Optional[float] = field(
       default=0,
       metadata={"help": "whether to omit outputs that don't fit in length context or not"},
    )
    len_penalty: Optional[float] = field(
       default=0,
       metadata={"help": "whether to omit outputs that don't fit in length context or not"},
    )
    scale_reward: Optional[float] = field(
       default=0,
       metadata={"help": "whether to omit outputs that don't fit in length context or not"},
    )
    trl_weird: Optional[float] = field(
       default=0,
       metadata={"help": "whether to omit outputs that don't fit in length context or not"},
    )
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="checkpoints/debugging", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=1, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=10000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2, #HACK used to be 0.2, make sure to switch back at some point
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    len_only: Optional[float] = field(
       default=0,
       metadata={"help": "whether to omit outputs that don't fit in length context or not"},
    )
    gen_bsize: Optional[int] = field(
       default=1,
       metadata={"help": "how many outputs to over-generate per sample"},
    )
    # these are some parameters for custom rollouts
    rollout_strategy: Optional[str] = field(default="normal", metadata={"help": "rollout strategy, start with high var, high mean, etc"})
    oversample: Optional[int] = field(
       default=1,
       metadata={"help": "how many outputs to over-generate per sample"},
    )
    log_genaccs: Optional[int] = field(
       default=1,
       metadata={"help": "how many outputs to over-generate per sample"},
    )
    temperature: Optional[float] = field(
       default=1,
       metadata={"help": "sampling temperature for generation"},
    )
    generators_json: Optional[str] = field(default=None, metadata={"help": "json file indicating which checkpoints to use for rollouts at various points"})
    # token count-based bonus for exploration, TODO will need to mess with this more, may be additional fun stuff to try
    tok_bonus_ratio: Optional[float] = field(
       default=0,
       metadata={"help": "bonus reward (token-based) to encourage unexplored tokens showing up more"},
    )
    keep_long: Optional[int] = field(
       default=0,
       metadata={"help": "how many outputs to over-generate per sample"},
    )
    replay_freq: Optional[int] = field(
       default=1000000,
       metadata={"help": "how many outputs to over-generate per sample"},
    )
    replay_updates: Optional[int] = field(
       default=0,
       metadata={"help": "how many batches to dedicate to replay once we start an update"},
    )
    replay_batches: Optional[int] = field(
       default=3,
       metadata={"help": "how big of a buffer to keep (multiplied with batch size) for resampling with replays"},
    )