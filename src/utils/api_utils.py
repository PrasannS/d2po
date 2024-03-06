from utils.eval.rewards import get_synth_rewards
from utils.data.dataproc import append_dict_to_jsonl
import numpy as np
import random

"""
Given a set of rewards, / inputs, score everything with the gold function, update relevant data stuff

TODO clean up call code and expand the functionality a bit
"""
def get_gold_and_log(rewards, inps, input_texts, tokenizer, reward_model, script_args, i, metrics):
    allngs = []
    acc = 0
    totnew = 0
    # we're doing some kind of active update
    if script_args.tracking: 
        inds = list(range(0, len(rewards), 2))
        # can use either random-based or slightly more complex confidence thing for selecting examples to use
        if script_args.relab_criteria=="conf":
            diffs = list([abs(rewards[i]-rewards[i+1]) for i in range(0, len(rewards), 2)])
            inds = list([inds[i] for i in np.argsort(diffs)])
        elif "rand" in script_args.relab_criteria:
            random.shuffle(inds)
        inds = inds[:int(len(inds)*script_args.relabel_ratio)]
        
    for ind in range(0, len(rewards), 2):
        newgs = get_synth_rewards(input_texts[i+ind:i+ind+2], script_args.goldreward)
        tmp = {
            'texts':input_texts[i+ind:i+ind+2],
            'rewards':[float(f) for f in rewards[ind:ind+2]],
            'golds':newgs,
            # 'thresh':0,
            'labelled':metrics['label_count'],
            'step':metrics['call_count']
        }
        if newgs[0]!=newgs[1]:
            acc += 1 if ((newgs[0]>newgs[1])==(rewards[ind]>rewards[ind+1])) else 0
            totnew +=1
        # TODO this needs to have an abs on it I think? Currrently most examples are getting used
        # NOTE this used to be a thresholding thing, now base stuff on selection criteria code (above)
        if ind in inds:
            metrics['label_count']+=1
            if metrics['label_count']>=script_args.stopupdates:
                # NOTE if we label past a certain limit then stop grad updates
                script_args.noupdates=True
                reward_model.eval()
            # tmp['thresh'] = float(script_args.labelthresh*metrics['threshsum'])
            # TODO this may have been a big bug?
            allngs.append(newgs)
            # track how much extra data is needed (TODO at the beginning this will make some noise)
            if newgs[0]>newgs[1]:
                jind = ind
                kind = ind+1
            elif newgs[1]>newgs[0]:
                jind = ind+1
                kind = ind
            else: 
                append_dict_to_jsonl(tmp, script_args.logfile, metrics['logdata'])
                continue
            if script_args.tracking:
                # create new dataset on the fly
                metrics['extradata'].insert(0, {"input_ids_j":inps.input_ids[jind], "attention_mask_j":inps.attention_mask[jind],
                                    "input_ids_k":inps.input_ids[kind], "attention_mask_k":inps.attention_mask[kind]})
                # keep track of how much each datapoint gets reused
                keyval = tokenizer.decode(inps.input_ids[jind], skip_special_tokens=True)+tokenizer.decode(inps.input_ids[kind], skip_special_tokens=True)
                if keyval not in metrics['reuses']:
                    metrics['reuses'][keyval] = 0
                metrics['reuses'][keyval] = metrics['reuses'][keyval]+1
        append_dict_to_jsonl(tmp, script_args.logfile, metrics['logdata'])
        
    print(len(allngs)," new rewards: ", allngs)
    if totnew>0:
        print("acc is", acc/totnew)