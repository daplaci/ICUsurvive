#!/usr/bin/python
import concurrent.futures
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, pyll, trials_from_docs
from hyperopt.base import SONify
import hyperopt
import os, itertools, time, shutil, sys, stat, subprocess, pdb, random
import utils
from threading import Thread
import time, pickle
import pandas as pd
import numpy as np
from functools import partial
import multiprocessing 
import sys
import os, hashlib

CMD = '''#!/bin/bash
'''
path = os.getcwd()
nnet_file = "main.py"

#here the additional parameter that are not included in the grid search are specified
args = utils.parse_args()
moab_params = [k for argv in sys.argv[1:] for k in args.__dict__.keys() if k in argv]
additional_dict = {k: args.__dict__[k] for k in moab_params}

param_dict = {}
param_dict['n_window'] = '1-7-14-30-90-365'
param_dict['batch'] = 128

param_dict.update(additional_dict)

search_space = {
    'recurrent_layer' : hp.choice('recurrent_layer', ['lstm']),  #
    "diag_level" : hp.choice('diag_level', ['block', 'l3']),
    'units' : hp.randint('units', 32, 256),                      #
    'l2_regularizer' : hp.uniform('l2_regularizer', 1e-5, 1e-2),
    'embedding_coeff' : hp.uniform('embedding_coeff', 0.1, 2),   #
    'dropout': hp.uniform('dropout', 0.1,0.9),
    'recurrent_dropout': hp.uniform('recurrent_dropout', 0.1,0.9),
    'padd_percentile' : hp.randint('padd_percentile', 90, 98),
    'optimizer' : hp.choice('optimizer', ['Adagrad', 'adam'])
}

AUCheader = open('AUC_history_gridsearch.tsv', 'r').readlines()[0].strip().split('\t')

params_missing_AUCtable = [p for p in {**param_dict, **search_space}.keys() if p not in AUCheader]
print ("Adding : {} , as these columns were missing".format(params_missing_AUCtable))
AUCheader.extend(params_missing_AUCtable)
with open('AUC_history_gridsearch.tsv', 'w') as auc_file:
    auc_file.write('\t'.join(AUCheader) + '\n')

def optim_metric(dynamic_args):
    #generate flag string for the args your are tuning
    flag_dict = {**param_dict, **dynamic_args}
    param_string = '.'.join(['{}-{}'.format(k, flag_dict[k]) for k in sorted(list(flag_dict.keys()))])
    exp_id = hashlib.md5(param_string.encode()).hexdigest()
    flag_dict['exp_id'] = exp_id

    flag_string =  ''
    for k,v in flag_dict.items():
        flag_string += '--{} {} '.format(str(k),str(v))

    filename = '.'.join(['{}-{}'.format(k, dynamic_args[k]) for k in dynamic_args] +['sh'])
    shell_arg = "python -u {} {}".format(nnet_file, flag_string)
    
    shellfile = open('moab_jobs/{}'.format(filename), 'w')
    shellfile.write(CMD)
    shellfile.write('cd ' + path + '\n')
    shellfile.write(shell_arg + '\n')
    shellfile.close()
    st = os.stat('moab_jobs/{}'.format(filename))
    os.chmod('moab_jobs/{}'.format(filename), st.st_mode | stat.S_IEXEC)
    
    subprocess.call("./moab_jobs/{} > logs/{}.log 2>&1".format(filename, filename[:-3]), shell=True)
    print("Launched exp: {}\n".format(filename))

    time.sleep(5)

    print ("Reading AUC table for experiment id {} .. \n".format(exp_id))
    result_table = pd.read_csv('CV_history_gridsearch.tsv', sep='\t')

    subset = result_table[result_table.exp_id == exp_id]
    #assert len(subset) == args.n_splits*(len(param_dict['n_window'].split('-')))
    #optimization_param = 1 - np.mean(subset.auc_val.values)
    optimization_param = np.amin(subset.val_loss.values)
    return optimization_param

trial = Trials()
max_evals = 150
trials_file = "trials_baseline{}.pkl".format(param_dict["baseline_hour"])

while True:
    if trials_file in os.listdir('best_weights/'):
        trial = pickle.load(open('best_weights/{}'.format(trials_file),'rb'))
    
    best = fmin(optim_metric, search_space, algo=hyperopt.tpe.suggest, 
        max_evals=(trial.__len__() +1), trials=trial, show_progressbar=False)

    trial.refresh() 

    with open('best_weights/{}'.format(trials_file), 'wb') as f:
        pickle.dump(trial, f)

    if trial.__len__()>max_evals:
        break