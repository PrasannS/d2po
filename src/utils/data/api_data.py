import random 
from utils.data.dataproc import append_dict_to_jsonl
# TODO import logic might not work? 

"""
Given a datapoint from earlier (sdata), and later (edata) in training, 
create a new datapoint that will make a preference pair across both of them

# TODO maybe add some sort of thing so that the context is the same? 
"""
def crosstrain_pair(sdata, edata, tokenizer, device):
    # worse one from sdata
    sind = 0 if sdata['rewards'][0]<sdata['rewards'][0] else 1
    eind = 0 if edata['rewards'][0]>edata['rewards'][0] else 1
    # in case the
    inpk = tokenizer(sdata['texts'][sind], padding=True, truncation=True, return_tensors="pt").to(device)
    inpj = tokenizer(edata['texts'][eind], padding=True, truncation=True, return_tensors="pt").to(device)
    
    return {"input_ids_j":inpj.input_ids[0], "attention_mask_j":inpj.attention_mask[0],
                                            "input_ids_k":inpk.input_ids[0], "attention_mask_k":inpk.attention_mask[0]}
    
def trainheuristic_data(metrics, tokenizer, script_args, device):
    lasts = []
    starts = []
    if metrics['call_count']>script_args.heursteps:
        for i in range(len(metrics['call_count'])):
            if (metrics['logdata'][i]['step']==metrics['call_count']-script_args.heursteps):
                starts.append(metrics['logdata'][i])
            elif (metrics['logdata'][i]['step']==metrics['call_count']):
                lasts.append(metrics['logdata'][i])
    # get up to 20 combos via this formula
    for i in range(min(len(starts)*len(lasts), 20)):
        tmppair = crosstrain_pair(random.choice(starts), random.choice(lasts), tokenizer, device)
        metrics['extradata'].insert(0, tmppair)
        keyval = tokenizer.decode(tmppair['input_ids_j'], skip_special_tokens=True)+tokenizer.decode(tmppair['input_ids_k'], skip_special_tokens=True)
        if keyval not in metrics['reuses']:
            metrics['reuses'][keyval] = 0
        metrics['reuses'][keyval] = metrics['reuses'][keyval]+1

"""
Given a learning step, use heuristics to figure out what data could get re-used 
"""
def reuse_batchdata(ndiff, tokenizer, batch, metrics, readd_inds, script_args, cind, rewards_j, rewards_k):
    for i in range(len(ndiff)): 
        tj = tokenizer.decode(batch["input_ids_j"][i], skip_special_tokens=True)
        tk = tokenizer.decode(batch["input_ids_k"][i], skip_special_tokens=True)
        # second part of if is so we don't keep re-using same stuff in the generic baseline
        if (ndiff[i]<(script_args.labelthresh*metrics['threshsum'])) and (script_args.labelthresh<1): 
            readd_inds.append(cind+i)
            metrics['reuses'][tj+tk]+=1
        else: 
            tmp = {
                'texts':[tj, tk],
                'reuses':metrics['reuses'][tj+tk],
                'rewards':[float(rewards_j[i].detach()),float(rewards_k[i].detach())],
                'thresh':float(script_args.labelthresh*metrics['threshsum']), 
                'step':metrics['call_count'],
            }
            append_dict_to_jsonl(tmp, script_args.logfile)