
from statistics import mean
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import argparse
import json
# from rlhfutils.eval_utils import scofile
from nltk import word_tokenize
from utils.data.prompt_utils import splitter, convert_prompstlye, qaform
import utils.eval.rewards as rutils

toker = AutoTokenizer.from_pretrained("facebook/opt-125m")

def sconoundf(df, function, trunc=True):
    rlen=0
    allins = []
    # process all inps to run this stuff in a single batch
    for resps in df['response']:
        if len(resps)==4:
            rlen=4
            allins.extend([tokenproc(r, trunc, function) for r in resps])
        else:
            rlen=1
            allins.append(tokenproc(resps, trunc, function))
            # means.append(mean(get_synth_rewards([tokenproc(resps, trunc, function)], function) ))
    # for ins in allins:
    #     print("*****************")
    #     print(ins)
    rewards = rutils.get_synth_rewards(allins, function)
    means = []
    rets = []
    for i in range(0, len(allins), rlen): 
        means.append(mean(rewards[i:i+rlen]))
        rets.append(rewards[i:i+rlen])
    print(means)
    print(mean(means))
    return rets, mean(means)

def tokenproc(inp, lim=True, function=None):
    #print(inp)
    #print(function)
    if function=="math" or function=="contrastivedistill":
        return inp
    else:
        inp = convert_prompstlye(inp, qaform)
        
    if (function=="eurusrm") or (function=="paraphrase"):
        return inp
    
    if (function is None) or "contpos" not in function:
        inp = splitter(inp)[1].strip() # edited this code
        start=0
    else:
        # TODO edited this code, also it seems a bit strange
        start = len(toker(splitter(inp)[0].strip()).input_ids)
    if lim:
        tokd = toker(inp).input_ids[:start+50]
    else: 
        tokd = toker(inp).input_ids
    return toker.decode(tokd, skip_special_tokens=True)

# TODO why are things separate for bow, nouns? TODO will this work for tulu-format outputs?
def scofile(fname, gfunct, trunc=True, logind=0):
    idf = pd.read_json(fname, orient='records', lines=True)
    glens = []
    for i, row in idf.iterrows(): 
        if len(row['response'])<8:
            glens.append(len(word_tokenize(row['response'][0])) - len(word_tokenize(row['question'][0])))
        else:
            glens.append(len(word_tokenize(row['response'])) - len(word_tokenize(row['question'])))
        
    return sconoundf(idf, gfunct, trunc), glens

if __name__=="__main__": 
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Process two string arguments.')

    # Add arguments
    parser.add_argument('--fname', type=str, required=True, help='The filename as a string')
    parser.add_argument('--gfunct', type=str, required=True, help='The function name as a string')
    
    # Execute the parse_args() method
    args = parser.parse_args()
    
    # NOTE special case for distillation reward
    if "contrastivedistill" in args.gfunct:
        print("cdist case")
        rutils.likemod = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", device_map=0, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval()
        rutils.liketok = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
        rutils.slikemod = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map=0, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval()
        rutils.sliketok = AutoTokenizer.from_pretrained("facebook/opt-125m")
    if "eurusrm" in args.gfunct:
        print("eurus RM loading")
        # RM with its own custom code setup
        rutils.slikemod = AutoModel.from_pretrained("openbmb/Eurus-RM-7b", device_map=0, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        rutils.sliketok = AutoTokenizer.from_pretrained("openbmb/Eurus-RM-7b")
        
    t, lens = scofile(args.fname, args.gfunct, True, 0)
    fv, lens = scofile(args.fname, args.gfunct, False, 0)
    tval, tmeans = t
    fval, fmeans = fv
    
    # use everything except the filename
    outf = args.fname.replace(".jsonl", "")+".results"
    with open(outf, 'w') as f:
        json.dump({
            'truncval':tval,
            'notruncval':fval,
            'meanlen':mean(lens),
            'truncdist':tmeans,
            'notrunctdist':fmeans,
            'lendist':lens
        }, f)