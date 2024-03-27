import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('../src/')
from utils.data.prompt_utils import qaform

from statistics import mean

# get sliding window means over some input list (smoothened)
def intervmean(vals, interv=50, window=400):
    return [mean(vals[i:i+window]) for i in range(0, len(vals)-window, interv)]    

def combineall(vallists, inte=50, wind=400):
    if type(vallists[0][0])==list:
        newls = [[mean(m) for m in blist] for blist in vallists]
    else:
        newls = vallists
    finl = []
    for i in range(len(newls[0])):
        finl.append(mean([newls[j][i] for j in range(len(newls))]))
    return finl
    

def loadf(fname, gfunct=None, useself=False): 
    tlog = pd.read_json(fname, orient='records', lines=True)
    tmp = tlog
    print(tmp.keys())
    print(fname)
    if 'golds' not in tmp.keys():
        # no golds, rollo 
        if gfunct != None:
            print("getting some stuff")
            tgolds = []
            for _, row in tqdm(tmp.iterrows()):
                # NOTE need to set this up as a lambda (so that global var is set correctly)
                tgolds.append(gfunct([qaform(row['inputs'][i], row['outputs'][i]) for i in range(len(row['inputs']))]))
            tmp['golds'] = tgolds
        else:
            print("not getting some stuff")
            tmp['golds'] = tmp['rewards']
    if "selfreward" in fname:
        print("selfreward overrides")
        tmp['golds'] = [g[:4] for g in tmp['golds']]
    if useself: 
        tmp['rewards'] = tmp['selfscos']
    tmp = tmp.dropna(subset='golds')
    print(len(tmp))
    return tmp


def plot_methods(methods, steps=2000, fname="output.pdf", setname="Experiment Results", methmax=True, xlabel='X-axis Label', ylabel='Y-axis Label', dpoline=-10):
    # Set the aesthetic style of the plots
    sns.set(style="whitegrid")
    
    # Initialize variables for max_x_value calculation if methmax is True
    max_x_values = []

    colors = ['red', 'blue', 'green']

    cind = 0

    # Creating a figure and axis object
    plt.figure(figsize=(10, 6))
    
    # Loop through each method to plot
    for label, (data, ratio) in methods.items():
        # Normalize the x-axis values
        x_values = np.linspace(0, steps, len(data)) * ratio
        max_x_values.append(max(x_values))
        
        # Plot the method
        sns.scatterplot(x=x_values, y=data, label=label, s=75, edgecolor=colors[cind])

    if methmax:
        # Determine the smallest max_x_value across all methods if required
        max_x_value = min(max_x_values)
        
        # Clear the plot to redraw it with limited x values
        plt.clf()
        plt.figure(figsize=(10, 6))
        
        # Redraw each plot with limited x values
        for label, (data, ratio) in methods.items():
            x_values = np.linspace(0, steps, len(data)) * ratio
            filtered_x_values, filtered_data = zip(*[(x, y) for x, y in zip(x_values, data) if x <= max_x_value])
            sns.scatterplot(x=filtered_x_values, y=filtered_data, label=label, s=75, color=colors[cind], edgecolor=colors[cind])
            cind+=1

    if dpoline>-1:
        plt.axhline(y=dpoline, linestyle='dashed', color='black')


    # Enhancements for clarity and aesthetics
    plt.title(setname, fontsize=26, fontweight='bold')
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.legend(prop={'size': 16})
    plt.tight_layout()
    
    # Saving the figure in high-quality
    plt.savefig(fname, format="pdf", bbox_inches="tight")

    plt.show()


# Function to create a scatter plot comparing a baseline method to a new method
def ratio_scatter(baseline, method, steps=2000, baseratio=1, methodratio=1, fname="output.pdf", setname="Experiment Results", methmax=True, labels=['OPO with Gold', "ARMoUR"]):
    # Set the aesthetic style of the plots to be more appealing and professional
    sns.set(style="whitegrid")

    # Normalize the x-axis values for each list
    baseline_x = np.linspace(0, steps, len(baseline)) * baseratio
    method_x = np.linspace(0, steps, len(method)) * methodratio

    if methmax:
        # Determine the max x-axis value based on the smaller dataset
        max_x_value = min(max(baseline_x), max(method_x))
        
        # Filter out the points that exceed the max_x_value for both datasets
        baseline_x, baseline = zip(*[(x, y) for x, y in zip(baseline_x, baseline) if x <= max_x_value])
        method_x, method = zip(*[(x, y) for x, y in zip(method_x, method) if x <= max_x_value])

    # Creating a figure and axis object
    plt.figure(figsize=(10, 6))

    # Plot each dataset using seaborn for better aesthetics
    if baseratio>=0:
        sns.scatterplot(x=baseline_x, y=baseline, color='blue', label=labels[0], s=75)
    sns.scatterplot(x=method_x, y=method, color='red', label=labels[1], s=75, edgecolor='red')

    # Adding enhancements for clarity and aesthetics
    plt.title(setname, fontsize=26, fontweight='bold')
    plt.xlabel('Gold Preference Data Used', fontsize=20)
    plt.ylabel('Average Gold Reward' if baseratio>=0 else "RM Accuracy on Samples", fontsize=20)
    plt.legend(prop={'size': 16})
    plt.tight_layout()

    # Saving the figure in high-quality
    plt.savefig(fname, format="pdf", bbox_inches="tight")

    plt.show()

def makengs(ntmp, sind=0): 
    rat = 0
    ngs = []
    accs = []
    sind = 0
    print(ntmp.loc[0])
    for ind, row in ntmp.iloc[sind:].iterrows():
        # row['golds'] = get_synth_rewards(row['texts'], 'bagofwords')
        if len(row['golds'])==0:
            continue
        ngs.append(row['golds'])
        if row['golds'][0]==row['golds'][1]:
            accs.append(mean(accs[-10:]) if len(accs)>0 else 0.5)
            continue
        if ((row['rewards'][0]>row['rewards'][1])!=(row['golds'][0]>row['golds'][1])):
            rat+=1
            accs.append(0)
        else:
            accs.append(1)
    #tmp['golds'] = ngs
    print(rat/len(ngs))
    return ngs, accs

def accscatter(acclist, interv=200): 
    vals = []
    for j in range(0, len(acclist), interv):
        vals.append(mean(acclist[j:j+interv]))
    return plt.scatter(range(len(vals)), vals)
    