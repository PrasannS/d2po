from utils.eval.rewards import get_synth_rewards
import utils.eval.rewards as rw
from utils.data.dataproc import append_dict_to_jsonl
import numpy as np
import random
from statistics import mean

"""
Given a set of rewards, / inputs, score everything with the gold function, update relevant data stuff

TODO clean up call code and expand the functionality a bit
"""
def get_gold_and_log(inds, tokenizer, script_args, metrics):
    # special case for this thing
    if "contdist" in metrics.keys():
        rw.likemod = metrics['contdist'][0]
        rw.liketok = metrics['contdist'][1]
        rw.slikemod = metrics['contdist'][2]
        rw.sliketok = metrics['contdist'][3]
        
    allngs = []
    acc = 0
    totnew = 0
    
    for ind in range(0, len(metrics['rscores']), 2):
        # TODO just make this a script_arg, maybe have a hard limit for the ultra version? 
        getgold = (ind in inds) if ("ultra" in script_args.goldreward) else True
        # to prevent going bankrupt
        newgs = get_synth_rewards(metrics['all_texts'][ind:ind+2], script_args.goldreward) if getgold else None
        allngs.append(newgs)
        
        if (len(newgs)>0) and (newgs[0]!=newgs[1]):
            totnew +=1
            acc += 1 if (newgs[0]>newgs[1]) == (metrics['rscores'][ind]>metrics['rscores'][ind+1]) else 0
                

        tmp = {
            'texts':metrics['all_texts'][ind:ind+2],
            'rewards':[float(f) for f in metrics['rscores'][ind:ind+2]],
            'golds':newgs,
            'call':metrics['cinds'][ind:ind+2]
        }
        append_dict_to_jsonl(tmp, script_args.logfile, metrics['logdata'])
        # Add to data now
        if ind in inds:
            # track how much extra data is needed (TODO at the beginning this will make some noise)
            if newgs[0]>newgs[1]:
                jind = ind
                kind = ind+1
            elif newgs[1]>newgs[0]:
                jind = ind+1
                kind = ind
            else: 
                continue
            
            
            if script_args.tracking:
                # create new dataset on the fly
                metrics['extradata'].insert(0, {"input_ids_j":metrics['inpids'][jind], "attention_mask_j":metrics['masks'][jind],
                                    "input_ids_k":metrics['inpids'][kind], "attention_mask_k":metrics['masks'][kind]})
                # keep track of how much each datapoint gets reused
                keyval = tokenizer.decode(metrics['inpids'][jind], skip_special_tokens=True)+tokenizer.decode(metrics['inpids'][kind], skip_special_tokens=True)
                if keyval not in metrics['reuses']:
                    metrics['reuses'][keyval] = 0
                metrics['reuses'][keyval] = metrics['reuses'][keyval]+1
        
        
    print(len(allngs)," new rewards: ", mean([mean(m) for m in allngs]))
    if totnew>0:
        print("acc is", acc/totnew)