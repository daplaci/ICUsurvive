import keras
import json
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Embedding , LSTM, Input, Concatenate, Reshape
from keras.layers import Lambda,AveragePooling2D
from keras import backend as K
from tqdm import tqdm
import pdb,os, pickle
import utils
import learning
import data_utils
from p_tqdm import *
from pathos.multiprocessing import ProcessingPool
from multiprocessing import Pool
import writer
import sys
import create_y_train

args = utils.parse_args()

#DEFINE FUNCTIONS
#function to sort the time step as a list
sort_time = lambda x : list(map(str, sorted(list(map(float,x)), reverse=False)))
#this function is used to compute the embedding size according to the formula mentioned in the paper 
comput_emb_size = lambda x: int(round(np.sqrt(np.sqrt(x))*6*args.embedding_coeff))

#this Class manage all the methods related to the features
class DomainClass():
    def __init__ (self,feat_name, train_json, val_json, pids, val_pids):   
        self.json_file = train_json
        self.val_json = val_json
        self.feat_name = feat_name
        if feat_name in padd_dict:
            self.padd_size = padd_dict[feat_name]
        else:
            self.padd_size = 1
        self.padd_time = padd_dict['rel_time']
        self.tokenizer = keras.preprocessing.text.Tokenizer(num_words=None,filters='\t\n', lower=True,split=' ', char_level=False, oov_token=1)
        self.vocab = []
        self.inputs = []
        self.pids = pids
        self.val_pids = val_pids
        self.survivals = survivals
        self.pids_to_static = pids_to_static

    def build_vocab(self):
        if 'rel_time' in self.feat_name:
            [self.vocab.append(t) for p in self.json_file.keys() for t in self.json_file[p].keys() if float(t) < args.baseline_hour]
        else:
            for p in self.json_file.keys():
                for time in self.json_file[p].keys():
                    if self.feat_name in self.json_file[p][time] and float(time) < args.baseline_hour:
                        self.vocab.extend(self.json_file[p][time][self.feat_name])

        self.vocab = set(self.vocab) #this makes thing faster but you do not have estimate on frequency

    def fit_tokenizer(self):

        if self.feat_name in json_count:
            low_count_words = [w for w,c in json_count[self.feat_name].items() if int(c) < args.min_word_count and w in self.vocab]
            print ("Remove # {}low count words for {}.. ".format(len(low_count_words), self.feat_name), end='\r')
            for w in low_count_words:
                self.vocab.remove(w)

        self.tokenizer.fit_on_texts(self.vocab)

        self.vocab_size = len(self.tokenizer.word_index)+1
        if self.feat_name == 'rel_time':
            self.emb_size = min(10, comput_emb_size(len(self.tokenizer.word_index)))
        else:
            self.emb_size = comput_emb_size(len(self.tokenizer.word_index))
        print ("Embedding size for input {} is: {} \t Vocab size of dictionary : {}".format(self.feat_name, self.emb_size, self.vocab_size))

    def get_time_token(self, json_file,  patient):    
        time_text = sort_time(json_file[patient].keys())
        time_text = [t for t in time_text if float(t) < args.baseline_hour and any([args.__dict__[d] for d in json_file[patient][t]])] #make sure that at least one domain for that time step is TRUE
        time_text_padd = self.padder(time_text, self.padd_time)

        if 'rel_time' not in self.feat_name:
            return time_text_padd, None
        else:
            time_token = [x[0] for x in self.tokenizer.texts_to_sequences(time_text)]
            time_token_padd = self.padder(time_token, self.padd_time)
            return None , time_token_padd
            
    def extract_data(self, dataset='train'):
        #using all patients here
        if dataset=='train':
            json_file = self.json_file
            pids = self.pids
        else:
            json_file = self.val_json
            pids = self.val_pids
        
        inputs = []
        for p in tqdm(pids):

            time_text, time_token = self.get_time_token(json_file, p)
            if 'rel_time' in self.feat_name:
                inputs.append(time_token)
                continue

            feat_patient = []
            for time in time_text:
                if time not in json_file[p].keys():
                    feat_patient.append(self.padder([],self.padd_size))
                    continue
                
                if self.feat_name in json_file[p][time].keys():
                    feat_token = [x[0] for x in self.tokenizer.texts_to_sequences(json_file[p][time][self.feat_name])]
                elif self.feat_name in pids_to_static:
                    feat_token = [pids_to_static[self.feat_name][p]]
                else:
                    feat_token = [self.tokenizer.word_index[1]]
                #padd diag
                feat_patient.append(self.padder(feat_token,self.padd_size))
                
            inputs.append(feat_patient)

        assert pids.shape[0] == len(inputs)
        print ("Input created for {} for the dataset {}".format(self.feat_name, dataset))
        return np.asarray(inputs)
    
    def save_dictionary(self):
        with open('best_weights/{}_dictionary.pkl'.format(self.feat_name), 'wb') as f:
            pickle.dump(self.tokenizer.word_index, f)
    
    def padder(self, x, max_len, tkn=0):
        return [tkn]*(max_len - len(x)) + x[-max_len:]

def parallel_extract_data(obj):
    return obj.extract_data()

def get_padd(json_file, pids, percentile):
    padd_dict = {}
    padd_dict['rel_time'] = []
    print ("Generating dictionary for deciding the padd..")
    for p in tqdm(pids):
        time_vect = [t for t in json_file[p].keys() if float(t) < args.baseline_hour and any([args.__dict__[d] for d in json_file[p][t]])] #make sure that at least one domain for that time step is TRUE
        padd_dict['rel_time'].append(len(time_vect))
        for t in json_file[p]:
            for k in json_file[p][t]:
                if k not in padd_dict:
                    padd_dict[k] = []
                padd_dict[k].append(len(json_file[p][t][k]))
    for k in padd_dict:
        padd_dict[k] = int(round(np.percentile(np.asarray(padd_dict[k]), percentile)))
    print (padd_dict)
    return padd_dict


############################################################################################################################################################
## LOAD JSON THAT INCLUDES THE FEATURES 
## if the features are in different json - you have to be sure that the time is the same - 
## otherwise when you merge the different features you will have in the same raw feature belonging to different time point) #######
############################################################################################################################################################
print ("Loading json.. ", end='\r')
json_file = json.load(open('/home/projects/icu/final_gum/jsons_surv/train_{}.json'.format(args.diag_level), 'r'))
json_count = json.load(open('/home/projects/icu/final_gum/jsons_surv/counts_train_{}.json'.format(args.diag_level), 'r'))
print ("Json Loaded")

filter_table = pd.read_csv("/home/projects/icu/final_gum/output/courseid_permission_regH.csv", sep='\t').astype(str)
courseid2permission = dict(zip(filter_table.courseid_unique, filter_table.perm_both))

for k in list(json_file.keys()):
    if courseid2permission[k] != '1':
        del json_file[k]
assert len(json_file) == 10498

if args.save_all:
    print ("Loading Validation json.. ", end='\r')
    json_val_file = json.load(open('/home/projects/icu/final_gum/jsons_surv/val_{}.json'.format(args.diag_level), 'r'))
    print ("Val Json Loaded")
    
    for k in list(json_val_file.keys()):
        if courseid2permission[k] != '1':
            del json_val_file[k]
    assert len(json_val_file) == 2753

else:
    json_val_file = None
    val_pids = None


##########################
##LOAD whatever file you use for getting the outcome of the patient###
##########################
print ("Loading lpr table..", end='\r')
lpr_table = pd.read_csv('/home/projects/icu/final_gum/jsons_surv/outcome.tsv', sep='\t')
print ("Lpr Table Loaded")
pids_to_failure_time = dict(zip(lpr_table.courseid_unique.astype(str), lpr_table.failure_time))
pids = np.asarray([p for p in json_file.keys() if pids_to_failure_time[p] > args.baseline_hour//24]) #filter the patient whose failure time is higher then the baseline
if args.max_num_patients:
    pids = pids[:args.max_num_patients] 

print ("Loading metadata..", end='\r')
metadata_table = pd.read_csv('/home/projects/icu/final_gum/metadata/metadata_20201114.tsv', sep='\t', usecols=['courseid_unique', 'los_before_icu', 'ageadm'])
print ("Metadata Table Loaded")
pids_to_los_before_icu = dict(zip(metadata_table.courseid_unique.astype(str), metadata_table.los_before_icu))
pids_to_age = dict(zip(metadata_table.courseid_unique.astype(str), metadata_table.ageadm))
pids_to_static = {'ageadm':pids_to_age, 'los_before_icu': pids_to_los_before_icu}

breaks = np.array([0] + [int(x) for x in args.n_window.split('-')])  # Make list of integers from e.g ['1-7-14-30-90'] - add 0
survivals = create_y_train.make_surv_array((lpr_table.failure_time - args.baseline_hour//24), lpr_table.failure, breaks)

pids_to_survival = dict(zip(lpr_table.courseid_unique.astype(str), survivals))
y_window = np.asarray([list(pids_to_survival[p]) for p in pids])

if args.save_all:
    val_pids = np.asarray([p for p in json_val_file.keys() if pids_to_failure_time[p] > args.baseline_hour//24]) #filter the patient whose failure time is higher then the baseline
    y_val_window = np.asarray([list(pids_to_survival[p]) for p in val_pids])

padd_dict = get_padd(json_file, pids, args.padd_percentile)

######################
## Initialize the different features###
######################
time_obj = DomainClass('rel_time', json_file, json_val_file, pids, val_pids)
ageadm_obj = DomainClass('ageadm', json_file, json_val_file, pids, val_pids) if args.ageadm else None
los_before_icu_obj = DomainClass('los_before_icu', json_file, json_val_file, pids, val_pids) if args.los_before_icu else None
diag_obj = DomainClass('diag', json_file, json_val_file, pids, val_pids) if args.diag else None
opr_obj = DomainClass('opr', json_file, json_val_file, pids, val_pids) if args.opr else None
ube_obj = DomainClass('ube', json_file, json_val_file, pids, val_pids) if args.ube else None
chart_obj = DomainClass('chart', json_file, json_val_file, pids, val_pids) if args.chart else None
medication_obj = DomainClass('medication', json_file, json_val_file, pids, val_pids) if args.medication else None
equipment_obj = DomainClass('equipment', json_file, json_val_file, pids, val_pids) if args.equipment else None
biochemical_obj = DomainClass('biochem', json_file, json_val_file, pids, val_pids) if args.biochem else None

#if domain is not None will be added in this list (the domain by default is True unless in the gridsearch you set it as False)
list_input_domain = [domain for domain in [time_obj, diag_obj, opr_obj, ube_obj, chart_obj, equipment_obj, medication_obj, biochemical_obj, ageadm_obj, los_before_icu_obj] if domain]
[feat.build_vocab() for feat in list_input_domain]
[feat.fit_tokenizer() for feat in list_input_domain]

args.cv_split = data_utils.init_cv_split(pids, np.array([pids_to_failure_time[p] for p in pids]), args)
args.padd_time = padd_dict['rel_time']

print ("Changing to dir:{}".format(args.output_dir))
os.chdir(args.output_dir)

df_writer = writer.DF_writer(pids, breaks)

#uncomment this if you want to use the dictionary - TODO: can be added as a flag
#[feat.save_dictionary() for feat in list_input_domain]
data_extracted = list(map(lambda x: x.extract_data(), list_input_domain))

for inp, feat, in zip(data_extracted, list_input_domain):
    feat.inputs = inp

if args.save_all:
    #save here the input for the model on training
    np.savez('data/input_training_data_percentile{}_baseline{}_embcoeff{}.npz'.format(args.padd_percentile, args.baseline_hour, args.embedding_coeff), *data_extracted)
    np.save('data/target_training_data_percentile{}_baseline{}_embcoeff{}.npy'.format(args.padd_percentile, args.baseline_hour, args.embedding_coeff), y_window)
    np.save('data/pids_training_data_percentile{}_baseline{}_embcoeff{}.npy'.format(args.padd_percentile, args.baseline_hour, args.embedding_coeff), pids)

    #save everithing for val
    val_data_extracted = list(map(lambda x: x.extract_data(dataset='val'), list_input_domain))
    np.savez('data/input_val_data_percentile{}_baseline{}_embcoeff{}.npz'.format(args.padd_percentile, args.baseline_hour, args.embedding_coeff), *val_data_extracted)
    np.save('data/target_val_data_percentile{}_baseline{}_embcoeff{}.npy'.format(args.padd_percentile, args.baseline_hour, args.embedding_coeff), y_val_window)
    np.save('data/pids_val_data_percentile{}_baseline{}_embcoeff{}.npy'.format(args.padd_percentile, args.baseline_hour, args.embedding_coeff), val_pids)

#    try:
print ("#### Training window:{}".format(args.window_list))
learning.train_model(list_input_domain, y_window ,args, df_writer)

#        df_writer.write_progress()
#    except:
#        df_writer.write_error()

