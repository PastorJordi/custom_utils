import sys
import os
import numpy as np
import pandas as pd
import swifter as sw
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
import tqdm
from ast import literal_eval
#if notebook:
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

def groupby_binom_ci(x):
    # so we can plot groupby with errorbars in binomial vars in 2 lines
    return [abs(x.mean()-ci) for ci in proportion_confint(x.sum(), len(x), method='beta')]

def get_datecol(df):
    """input df, returns"""
    if 'sessid' in df.columns:
            return pd.to_datetime(df.sessid.str[-15:])
    else:
        print('sessid not in df.cols')

common_mask = 'sound_len <= 400 and soundrfail == False and resp_len <=1 and R_response>= 0 and hithistory >= 0 and special_trial == 0'