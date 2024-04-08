# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from accelerate import Accelerator

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, AutoModelForCausalLM

from trl import DPOTrainer
import time
from rlhfutils.data import load_rlcd, load_wgpt, load_stack, inp_origformat, adjust_apf, load_manual
from utils.args.dpo_args import DPOArguments

os.environ["WANDB_TAGS"] = "[\"dporlhf\"]"

#HACK currently modified DPO logging code

def tulu_pf(question, answer):
    return "<user>\n"+question+"\n<assistant>\n"+answer

def onlyans(_, answer):
    return "" 

def simplecat(question, answer): 
    return question

def ans(question, answer):
    return question + "\nAnswer: \n"
    
def load_dpo_data(
    script_args,
    dataset: str = None,
    eval_dataset: str=None,
    num_proc=12,
) -> Dataset:   
    if len(eval_dataset)==0 or "/"==eval_dataset[-1]:
        eval_dataset=None
        print("no eval dataset used")
    # Load in data from the right datasets
    if dataset == 'wgpt':
        train_data, eval_data = load_wgpt()
        pfunct = adjust_apf
    elif dataset == 'stack': 
        train_data, eval_data = load_stack()
        pfunct = inp_origformat
    elif dataset == 'rlcd':
        train_data, eval_data = load_rlcd()
        pfunct = adjust_apf
        
    else: 
        train_data, eval_data = load_manual(dataset, "", testdir=eval_dataset)
        pfunct = adjust_apf
        
    if 'distil' in dataset:
        print("NOTE we're doing distillation, use responses directly")
        pfunct = onlyans
        
    # adjust the prompt style as needed
    # if "default" not in script_args.promptstyle: 
    #     pfuncts = {'onlyans': onlyans, 'dircat':simplecat, "ans":ans}
    #     pfunct = pfuncts[script_args.promptstyle]
    
    # TODO some conflicting things here
    if "tulu" in script_args.model_name_or_path:
        pfunct = tulu_pf

    # TODO add in a sanity check

    original_columns = train_data.column_names
    eval_columns = eval_data.column_names
    
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            # PFUNCT is variable based on the base model (question: answer: for stack, ##instruction when using apf base model)
            "prompt": [pfunct(question, "") for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    final_train = train_data.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )    
    final_eval = eval_data.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=eval_columns,
    )

    return final_train, final_eval

if __name__ == "__main__":
    parser = HfArgumentParser(DPOArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        device_map={"": Accelerator().local_process_index},
        load_in_8bit=False, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    )
    
    refmodel = None # AutoModelForCausalLM.from_pretrained(
    #     "facebook/opt-125m",
    #     device_map={"": Accelerator().local_process_index},
    #     load_in_8bit=False, 
    #     torch_dtype=torch.bfloat16, 
    #     attn_implementation="flash_attention_2"
    # )
    
    # model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # model_ref = AutoModelForCausalLM.from_pretrained(
    #     script_args.model_name_or_path,
    #     device_map={"": Accelerator().local_process_index},
    #     load_in_8bit=True
    # )
    # NOTE changed tokenizer path hardcoding
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # NOTE we're now doing this model
    train_dataset, eval_dataset = load_dpo_data(script_args, script_args.dataset, eval_dataset=script_args.evaldata)
    
    moreevals = []
    evlist = []
    if len(script_args.extraevaldata)>0:
        if "," in script_args.extraevaldata:
            evlist = script_args.extraevaldata.split(",")
        else:
            evlist = [script_args.extraevaldata]
        for e in evlist:
            _, etmp = load_dpo_data(script_args, e, eval_dataset=e)
            # use all the eval sets in comma list
            moreevals.append(etmp)
        print("ok we have extra datasets: ", evlist)
    
    
    print("length of dataset is ", len(train_dataset))
    
    print("TRAIN DATA")
    print(train_dataset[0]['prompt'])
    print(train_dataset[0]['chosen'])
    print(train_dataset[0]['rejected'])
    print("EVAL DATA")
    print(eval_dataset[0]['prompt'])
    print(eval_dataset[0]['chosen'])
    print(eval_dataset[0]['rejected'])
    
    # # 3. Load evaluation dataset
    
    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="trl_dpo",
        ddp_find_unused_parameters=False
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=refmodel,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        loss_type=script_args.loss_type,
    )
    # print("initial eval")
    # time.sleep(3)

    # print(dpo_trainer.evaluate())
    # time.sleep(3)
    # for e in range(len(evlist)):
        
    #     print("evaluating with", evlist[e])
    #     print(dpo_trainer.evaluate(moreevals[e]))
    #     print("done")
        
    # time.sleep(3)

    # 6. train
    dpo_trainer.train()
    
    # print("we're done")

    # print(dpo_trainer.evaluate())
    # time.sleep(3)
    # for e in range(len(evlist)):
        
    #     print("evaluating with", evlist[e])
    #     print(dpo_trainer.evaluate(moreevals[e]))

    # time.sleep(3)
    print("Saving last checkpoint of the model")
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)