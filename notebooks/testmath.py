import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from statistics import mean
import sys

def calculate_nll(input_string):
    # Tokenize the input string
    inputs = tokenizer(input_string, return_tensors='pt')
    
    # Get input_ids and move to the appropriate device
    input_ids = inputs['input_ids']
    device = model.device
    input_ids = input_ids.to(device)
    
    # Get the logits from the model
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    
    # Calculate the loss
    loss = outputs.loss
    
    # Convert the loss to NLL
    nll = loss.item()
    
    return nll

if __name__=="__main__":
    adapt_path = sys.argv[1] #"../checkpoints/math/mbestofnsft_sft_bonsft/final_checkpoint/"
    base = "outputs/models/math/randbigsft/"
    # load in adapter / model
    
    model = AutoModelForCausalLM.from_pretrained(
        base, load_in_8bit=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(base)
    if "orig" not in adapt_path:
        # Load the Lora model
        peft_config = PeftConfig.from_pretrained(adapt_path)
        model = PeftModel.from_pretrained(model, adapt_path)
    model.eval()
    
    # load in data to get likelihoods on
    bonds = pd.read_json("notebooks/nmath.json", lines=True, orient='records')
    
    glikes = []
    for i, r in tqdm(bonds.iterrows(), total=len(bonds)):
        tmplikes = [calculate_nll(r['ags'][ind]) for ind in range(len(r['ags']))]
        glikes.append(tmplikes)
    
    bonds['bonsft_likes'] = glikes

    bonds.to_json("notebooks/nmath"+str(sys.argv[2])+".json", lines=True, orient='records')