#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
import json


INPUT_DIR = '/home/projects/icu/gum/input/'

KEEP_ONLY_SET = True
AGG_HOURS = 24

print('LOADING CODES...')
data = pd.read_csv(INPUT_DIR + 'list_of_codes_before_adm.tsv' , sep='\t')
print('DONE')

#############################
## GENERATE LONG-FORMAT DF ##
#############################

# [name, code_column, timedelta_column]
feature_list = [['diag', 'C_DIAG', 'C_DIAG_timedelta_h'], 
                ['opr', 'C_OPR_x', 'C_OPR_timedelta_h_x'],
                ['ube', 'C_OPR_y', 'C_OPR_timedelta_h_y']]

row_list = []

print('GENERATES LONG-FORMAT...')
for index, row in tqdm(data.iterrows()):
    courseid_unique = row['courseid_unique']
    for feature in feature_list:
        if pd.isnull(row[feature[1]]) or pd.isnull(row[feature[2]]): continue  
        codes = ast.literal_eval(row[feature[1]])
        t_deltas = ast.literal_eval(row[feature[2]])
        
        for idx, code in enumerate(codes):
            #print(idx, code)
            row_dict = {}
            row_dict.update({'courseid_unique': courseid_unique, 
                            'timedelta': t_deltas[idx],
                            'feature': feature[0],
                            'code': code})
            row_list.append(row_dict)

data_long = pd.DataFrame(row_list) 
print('DONE')

# create 'time_window' 
# time_window 0: 0 < timedelta <= AGG_HOURS
# time_window 1: AGG_HOURS < timedelta <= 2*AGG_HOURS
data_long['time_window'] = np.floor(data_long['timedelta'] / AGG_HOURS)

# aggregate codes per 'time_window'
data_long_agg = data_long.groupby(['time_window', 'courseid_unique', 'feature'])['code'].apply(list).reset_index()


###################
## POPULATE DICT ##
###################

data_dict = {}

print('POPULATING DICT...')
for courseid in tqdm(data_long_agg.courseid_unique.unique().tolist()):
    
    df = data_long_agg.loc[data_long_agg.courseid_unique==courseid]
    timedeltas = df.time_window.unique().tolist()
    
    time_dict = {}
    for timedelta in timedeltas:
        df_time = df.loc[df.time_window == timedelta] 
        features = df_time.feature.unique().tolist()
        
        feature_dict = {}
        for feature in features:
            df_feature = df_time.loc[df_time.feature == feature]
            if KEEP_ONLY_SET:
                feature_dict[feature] = list(set(df_feature.iloc[0,3])) # NEED TO FIX ILOC
            else:
                feature_dict[feature] = df_feature.iloc[0,3]
            
        time_dict[timedelta] = feature_dict
    data_dict[courseid] = time_dict
print('DONE')    

print('SAVES JSON...')
with open(INPUT_DIR + 'data.json', 'w') as f:
    json.dump(data_dict, f)   
