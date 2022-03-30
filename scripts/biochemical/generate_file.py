import pandas as pd
import numpy as np
from date import parse_date
import os,sys
import multiprocessing
from tqdm import tqdm
from p_tqdm import *
import argparse
from multiprocessing import Process, Manager, Pool
import time
import itertools 
import warnings
import ast, pickle

warnings.filterwarnings('ignore')

def parse_args(args_str=None):
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(description='ICU Survival model')
    parser.add_argument('--bcc_path', type=str, default='/home/projects/icu/gum/input/bcc/bcc_data.tsv', help='bcc_path')
    parser.add_argument('--metadata_path', type=str, default='/home/projects/icu/gum/metadata/metadata_all.tsv', help='metadata_path')
    parser.add_argument('--labka_path', type=str, default='/home/projects/bth/biochemical/output/labka/measurement_var_len_cpr.tsv', help='labka path')
    parser.add_argument('--num_cores', type=int, default=150, help='number of cores to use in multiprocessing')
    parser.add_argument('--bcc_lines', type=int, default=36940944, help='number of lines in bcc')
    parser.add_argument('--labka_lines', type=int, default=263120628, help='number of lines in labka')
    parser.add_argument('--action', type=str, default='map', help='how to parallelize')
    parser.add_argument('--debug', type=str2bool, default=True, help='run in debug mode')
    
    #args for notes
    args = parser.parse_args()
    return args

def isDigit(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

args = parse_args()

#LOAD BCC INFO AND DATA
bcc_extract = pd.read_csv(args.bcc_path, sep='\t', nrows=5)
bcc_cols = bcc_extract.columns.tolist()

#LOAD LABKA DATA
labka_extract = pd.read_csv(args.labka_path, sep='\t', nrows=5)
labka_extract.columns = map(str.lower, labka_extract.columns)
labka_cols = labka_extract.columns.tolist()

#  load metadata
biochemical_meta = pd.read_csv(args.metadata_path, sep='\t', usecols=['courseid_unique', 'lpr_source','enc_cpr', 'cpr', 'admdatetime_utc'])
meta_cprs = biochemical_meta['cpr'].dropna().apply(lambda x : ''.join(str(x).split('-'))).unique().tolist()
meta_enc_cprs = biochemical_meta['enc_cpr'].dropna().astype(str).unique().tolist()
biochemical_meta['cpr'] = biochemical_meta['cpr'].apply(lambda x : ''.join(str(x).split('-')))
biochemical_meta['enc_cpr'] = biochemical_meta['enc_cpr'].apply(str)

remove_none = lambda x: [el for el in x if el is not None]
flatten = lambda x: [el for ls in x for el in ls]

with open('../../input/map_bcc_npu.pkl', 'rb') as f:
    mapping_dict = pickle.load(f)

unique_bcc_codes, unique_npu_codes, bcc_npu, npu_npu , npu_name = mapping_dict['unique_bcc_codes'], mapping_dict['unique_npu_codes'],\
                                                                mapping_dict['bcc_npu'], mapping_dict['npu_npu'], mapping_dict['npu_name']

def map_w_bcc(il):
    '''
        Process each line of the bcc file in order to generate a dict with courseid_unique , timestep and biochem value
        All the patients that are not ICU patients are left out now. Note that the mapping between KvntNr and NPU codes is not done at this level
        args:
            - il:tuple with line number and line
        returns:
            - dicts : contains for each courseid related to that enc_cpr a dictionary with courseid, timestep from admission in
            the ICU and biochemical value
    '''

    i,l = il
    print('{}/{}'.format(i,args.bcc_lines), end = '\r')
    values = l.strip('\n').split('\t')
    row_dict = dict(zip(bcc_cols, values))
    if row_dict['CPR'] not in meta_cprs:# or row_dict['KvntNr'] not in unique_bcc_codes:
        return None
    else:
        date = parse_date(row_dict['PrvtDatoTid'])  
        courseid_unique = biochemical_meta[biochemical_meta.cpr==row_dict['CPR']]['courseid_unique'].tolist()
        dicts = []
        for id_ in courseid_unique:
            admdate = parse_date(biochemical_meta[biochemical_meta.courseid_unique == id_].admdatetime_utc.iloc[0], convert_to_utc=False)
            timestep = (date - admdate).total_seconds() // 3600
            dict_to_df = {'courseid_unique':id_,
                          'timestep':timestep,
                         row_dict['KvntNr'] : row_dict['Resultat_tekst']}
            dicts.append(dict_to_df)
        return dicts

def map_w_labka(il):
    '''
        Process each line of the labka file in order to generate a dict with courseid_unique , timestep and biochem value
        All the patients that are not ICU patients are left out now
        args:
            - il:tuple with line number and line
        returns:
            - dicts : contains for each courseid related to that enc_cpr a dictionary with courseid, timestep from admission in
                    the ICU and biochemical value
    '''
    i,l = il 
    print('{}/{}'.format(i,args.labka_lines), end = '\r')
    values = l.strip('\n').split('\t')
    row_dict = dict(zip(labka_cols, values))
    
    if row_dict['pid'] not in meta_enc_cprs or not row_dict['numeric_value']:
        return None
    else:
        date = parse_date(row_dict['drawn_datetime_short'])
        courseid_unique = biochemical_meta[biochemical_meta.enc_cpr==row_dict['pid']]['courseid_unique'].tolist()
        dicts = []
        for id_ in courseid_unique:
            admdate = parse_date(biochemical_meta[biochemical_meta.courseid_unique == id_].admdatetime_utc.iloc[0], convert_to_utc=False)
            timestep = (date - admdate).total_seconds() // 3600
            dict_to_df = {'courseid_unique':id_,
                          'timestep':timestep,
                          row_dict['quantity_id'] : row_dict['numeric_value']}
            dicts.append(dict_to_df)
        return dicts

def build_dicts():
    '''
        Iterate over labka_to_df and bcc_to_df to build the dictionaries for the tsv's files
        args:
            - None
        returns:
            - l_dictionaries : is a dictionary, whose keys are the npu codes and their values are the dictionaries that will be used for
            generating the tsv files afterwards 

    '''
    l_dictionaries = {code:{'courseid_unique':[], 'timestep':[], code:[]} for code in npu_npu.values()}
    for it,d in enumerate(tqdm(bcc_to_df)):
        this_bcc = [ky for ky in d.keys() if ky not in ['courseid_unique', 'timestep']][0]
        d[this_bcc].replace(',', '.')
        if this_bcc in bcc_npu.keys() and isDigit(d[this_bcc]):
            this_npu = npu_npu[bcc_npu[this_bcc][0]]
        else:
            continue
        l_dictionaries[this_npu]['courseid_unique'].append(d['courseid_unique'])
        l_dictionaries[this_npu]['timestep'].append(d['timestep'])
        l_dictionaries[this_npu][this_npu].append(d[this_bcc])

    for it, d in enumerate(tqdm(labka_to_df)):
        # here can be tricky to understand. npu_npu is a dictionary that map multiple npu codes related to the same measurement
        # to a unique npu code. If npu0, npu1, and npu2 are all the codes of Kreatinin, the the dict will be {npu0:npu0, npu1:npu0, npu2:npu0}
        npu_code = [ky for ky in d.keys() if ky not in ['courseid_unique', 'timestep']][0]
        d[npu_code].replace(',', '.')
        if npu_code in npu_npu.keys() and isDigit(d[npu_code]):
            this_npu = npu_npu[npu_code]
        else:
            continue
        l_dictionaries[this_npu]['courseid_unique'].append(d['courseid_unique'])
        l_dictionaries[this_npu]['timestep'].append(d['timestep'])
        l_dictionaries[this_npu][this_npu].append(d[npu_code])

    return l_dictionaries

save_path = "../../input/biochemical/"

def dump_files(dictionary):
    '''
        Save a tsv file for each NPU code
        args:
            - dictionary: it is the dictionary related to a npu code that will be used for constructing the tsv 
        returns:
            - None - file is saved 

    '''
    df = pd.DataFrame.from_dict(dictionary)
    this_npu = [ky for ky in dictionary.keys() if ky not in ['courseid_unique', 'timestep']][0]
    assert type(this_npu) is str
    code_name = npu_name[this_npu]
    df = df.rename(columns={this_npu:code_name})
    try:
        os.makedirs(save_path)
    except:
        pass
    filename = "{}{}.tsv".format(save_path,this_npu)
    df.to_csv(filename, sep='\t', index=False)
 
#apply functions
if args.action == 'map_bcc':
    # process bcc
    with open(args.bcc_path, 'r') as f:
        if args.debug:
            head_f = [next(f) for n in range(1000)]
            bcc_to_df = [map_w_bcc((i,l)) for i,l in enumerate(tqdm(head_f))]
        else:
            bcc_to_df = Pool(args.num_cores).map(map_w_bcc, enumerate(f))
            print ("bcc ccompleted")

    bcc_to_df = remove_none(bcc_to_df)
    bcc_to_df = flatten(bcc_to_df)
    with open('../../input/dev_biochemical/bcc_to_df.pkl' , 'wb') as f:
        pickle.dump(bcc_to_df, f)

elif args.action == 'map_labka':
    #process labka
    with open(args.labka_path,'r') as f:
        if args.debug:
            head_f = [next(f) for n in range(1000)]
            labka_to_df = [map_w_labka((i,l)) for i,l in enumerate(tqdm(head_f))]
        else: 
            labka_to_df = Pool(args.num_cores).map(map_w_labka, enumerate(f))
            print ("labka completed")

    labka_to_df = remove_none(labka_to_df)
    labka_to_df = flatten(labka_to_df)
    
    with open('../../input/dev_biochemical/labka_to_df.pkl' , 'wb') as f:
        pickle.dump(labka_to_df, f)

elif args.action == 'dump_from_pickle':
    #save files
    bcc_to_df = pickle.load(open('../../input/dev_biochemical/bcc_to_df.pkl' , 'rb'))
    labka_to_df = pickle.load(open('../../input/dev_biochemical/labka_to_df.pkl' , 'rb'))

    print("generating dictionaries")
    l_dictionaries = build_dicts()  
    print("generating tsv files") 
    
    with open('../../input/dev_biochemical/all_codes_dict_to_df.pkl' , 'wb') as f:
        pickle.dump(l_dictionaries, f)
    
    for npu_code in tqdm(npu_npu.values()):
        dump_files(l_dictionaries[npu_code])
    sys.exit()
else:
    raise Exception("Unknown action: chose between 'map_bcc', 'map_labka' or  'dump_from_pickle'")
