import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, ttest_ind

# Test significance using bootstrap test as described in 
# "An Empirical Investigation of Statistical Significance in NLP"
# (Berg-Kirkpatrick et al., EMNLP 2012)
# https://aclanthology.org/D12-1091/
# i.e. based on "An Introduction to the Bootstrap" 
# (Efron and Tibshirani, 1993)
# m1_name is the column for System 1, similarly for m2_name
# Label column = "labels"
def bootstrap_micro(df, m1_name, m2_name, B):
    score = 0
    f1 = f1_score(df['labels'], df[m1_name], average='macro')
    f2 = f1_score(df['labels'], df[m2_name], average='macro')
    d_orig = f2 - f1
    print(f"\t# sentences {len(df)}")
    print(f"\td_orig = {d_orig:.4f}; f1 = {f1:.4f}, f2 = {f2:.4f}")
    n = len(df)
    
    set1 = list(zip(df['labels'], df[m1_name]))
    set2 = list(zip(df['labels'], df[m2_name]))
    bootstrap_dist = []
    
    for b in range(B):
        idx = np.random.choice(n, n)
        labels1, preds1, labels2, preds2 = [], [], [], []
        for i in idx:
            labels1.append(set1[i][0])
            preds1.append(set1[i][1])
            labels2.append(set2[i][0])
            preds2.append(set2[i][1])
        this_f1 = f1_score(labels1, preds1, average='macro')
        this_f2 = f1_score(labels2, preds2, average='macro')
        d_new = this_f2 - this_f1
        bootstrap_dist.append(d_new)
        if d_new > 0:
            score += 1
    b_mean = np.mean(bootstrap_dist)
    p_count = len([x for x in bootstrap_dist if x - b_mean >= d_orig])
    p_val = (p_count+1)/float(B+1)
    print(f"\t# of times Sys 2 is better = {score}, out of {B} bootstrap samples")
    print(f"\tp-val = {p_val:.4f}; bootstrap mean = {b_mean:.4f}")
    return p_val, 1 - score/B
