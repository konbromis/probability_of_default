# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:12:11 2019

@author: konstantinos
"""
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from itertools import groupby

def getMaxLength(x): 
    count = 0 
    # initialize max
    result = 0
    for i in x:
        if (i == 0): 
            count = 0
        else: 
    # increase count 
            count+= 1 
            result = max(result, count)         
    return result  

def getconsecmaxnochng(x):
    groups = groupby(x)
    result = [(label, sum(1 for _ in group)) for label, group in groups]
    if max(result, key=lambda x: x[1])[1] == 1:
        result2 = 0
    else:
        result2  = max(max(result, key=lambda x: x[1]))
    return result2

def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))