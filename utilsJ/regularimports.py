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
import scipy.io as spio
#if notebook:
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

def groupby_binom_ci(x, method='beta'):
    # so we can plot groupby with errorbars in binomial vars in 2 lines
    return [abs(x.mean()-ci) for ci in proportion_confint(x.sum(), len(x), method=method)]

def get_datecol(df):
    """input df, returns series"""
    if 'sessid' in df.columns:
            return pd.to_datetime(df.sessid.str[-15:])
    else:
        print('sessid not in df.cols')

common_mask = 'sound_len <= 400 and soundrfail == False and resp_len <=1 and R_response>= 0 and hithistory >= 0 and special_trial == 0'

lejla_rats = [f'LE{x}' for x in range(36,42)]
dani_rats =  [f'LE{x}' for x in range(42,48)]
jordi_rats = [f'LE{x}' for x in range(82,88)]

def apply_expand_concat(df,fun, newcolnames):
	"""returns apply+expand concatenated to df,
	fun: lambda x: func(row) should work too?
	newcolnames = list of newly generated columns
	"""
	tmp= df.swifter.apply(fun, axis=1, result_type='expand')
	tmp.columns = newcolnames
	return pd.concat([df, tmp], axis=1)


def set_share_axes(axs, target=None, sharex=False, sharey=False, visible_lab=False):
    """https://stackoverflow.com/questions/23528477/share-axes-in-matplotlib-for-only-part-of-the-subplots"""
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(visible_lab)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(visible_lab)

#https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
def loadmat(filename):
	'''
	this function should be called instead of direct spio.loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	'''
	data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
	return _check_keys(data)

def _check_keys(dict):
	'''
	checks if entries in dictionary are mat-objects. If yes
	todict is called to change them to nested dictionaries
	'''
	for key in dict:
		if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
			dict[key] = _todict(dict[key])
	return dict        

def _todict(matobj):
	'''
	A recursive function which constructs from matobjects nested dictionaries
	'''
	dict = {}
	for strg in matobj._fieldnames:
		elem = matobj.__dict__[strg]
		if isinstance(elem, spio.matlab.mio5_params.mat_struct):
			dict[strg] = _todict(elem)
		else:
			dict[strg] = elem
	return dict


def loadpickles(subjectlist, path='/home/jordi/Documents/changes_of_mind/firstanal_com_new_setup/all_data_so_far/'):
	"""because I have too much data, i'll simply load subjects and concatenate them from now on!"""
	df = pd.DataFrame([])

	pbar = tqdm.tqdm_notebook(subjectlist)
	for subj in pbar:
		pbar.set_description(subj)
		try:
			df = pd.concat(
				[df, pd.read_pickle(path+subj+'.pkl')], 
				ignore_index=True, sort=True)
		except Exception as e:
			print(f'error in {subj}\n{e}')
		
	df.reset_index(inplace=True, drop=True) 
	df.origidx =df.origidx.astype(int)
	print(f'returning a df with shape {df.shape}')
	return df	