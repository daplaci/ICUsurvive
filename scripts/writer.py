import traceback, os
import numpy as np
import pandas as pd
import data_utils, metrics
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Masking, CuDNNLSTM, GRU, CuDNNGRU, Bidirectional, Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from keras import backend as K
from copy import deepcopy
import hashlib
import nnet_survival
from sksurv.metrics import concordance_index_censored as ci

stripper = lambda x : str(x[0]) if ((type(x) is tuple or type(x) is list) and len(x)==1)  else str(x)


class DF_writer():
    def __init__(self, pids, breaks):
        self.AUCheader = list(pd.read_csv('AUC_history_gridsearch.tsv', sep='\t', nrows=1).columns.values)
        self.CVheader = list(pd.read_csv('CV_history_gridsearch.tsv', sep='\t', nrows=1).columns.values)
        self.progressHeader = list(pd.read_csv('progress.log', sep='\t', nrows=1).columns.values)
        self.errorHeader = list(pd.read_csv('error.log', sep='\t', nrows=1).columns.values)
        self.pids = pids
        self.breaks = breaks
        #self.id_col = 'courseid_unique' if 'courseid_unique' in self.metadata.columns else 'v_cpr_enc'
        #self.auc_cum_dict = {}
    
    def write_progress(self):
        print("Writing the progress of this Job")
        progressDF = pd.DataFrame(columns=self.progressHeader)
        progressDF = progressDF.append({'completed' : 'final'}, ignore_index = True)
        progressDF.to_csv('progress.log', header=None, index=False, 
                                    sep='\t', mode='a', columns = self.progressHeader)
        
    def write_error(self):
        print("\n*****Error.. Writing the error for this Job*****\n")
        var = traceback.format_exc()
        print (var)
        errorDF = pd.DataFrame(columns=self.errorHeader)
        errorDF = errorDF.append({'error': var,'args' : self.AUCheader}, ignore_index = True)
        errorDF.to_csv('error.log', header=None, index=False, sep='\t', mode='a', columns = self.errorHeader)
    
    def write_auc(self, args):
        args_to_df = {k:stripper(v) for k,v in args.__dict__.items() if k in self.AUCheader}
        print ("len of args  and aucheader : {}:{}".format(len(args_to_df), len(self.AUCheader)))
        #assert (len(args_to_df)==len(self.AUCheader))
        AUChistory = pd.DataFrame(columns=self.AUCheader)
        AUChistory = AUChistory.append(args_to_df, ignore_index = True)
        AUChistory.to_csv('AUC_history_gridsearch.tsv', index=False,sep='\t', mode='a', columns = self.AUCheader, header=None, float_format='%.3f')

    def write_history(self, args, train_hist):
        args_to_df = {k:stripper(v) for k,v in args.__dict__.items() if k in self.CVheader}
        train_hist.history.update(args_to_df)
        model_history = pd.DataFrame.from_dict(train_hist.history)
        model_history.to_csv('CV_history_gridsearch.tsv', index=True,sep='\t', mode='a', columns = self.CVheader, header=None, float_format='%.3f')
    
    def is_duplicate(self, args):
        args_to_df = {k:stripper(v) for k,v in args.__dict__.items() if k in self.AUCheader}
        AUChistory = pd.DataFrame(columns=list(args_to_df.keys()))
        AUChistory = AUChistory.append(args_to_df, ignore_index = True)
        AUCdataframe = pd.read_csv('AUC_history_gridsearch.tsv', usecols=list(args_to_df.keys()), sep='\t')
        AUCdataframe = AUCdataframe.drop_duplicates(keep='first')
        AUCtest_df = pd.concat([AUChistory, AUCdataframe], join='inner')
        if AUCtest_df.duplicated().any():
            print ("This args config already exists:\n", AUCtest_df.loc[AUCtest_df.duplicated(keep='first')].values)
        return AUCtest_df.duplicated().any()

    def update_train_results(self, X_train, y_train, X_test, y_test, model, filepath, train_hist, df_writer, args):                            
        try:
            #reload best weights
            model.load_weights(filepath)
            # calculate model prediction classes    
            y_pred_train = np.cumprod(model.predict(X_train), axis=1)
            y_pred_test = np.cumprod(model.predict(X_test), axis=1) 
            
            y_pred_train_binary = (y_pred_train > 0.5).astype('int32')
            y_pred_test_binary = (y_pred_test > 0.5).astype('int32')
            
            args_to_writer = deepcopy(args)
            for idx, window in enumerate(args.window_list):
                #add here all the args you want to write in the tables
                args_to_writer.n_window = window
                args_to_writer.auc_train = roc_auc_score(y_train[:, idx], y_pred_train[:, idx])
                args_to_writer.auc_val = roc_auc_score(y_test[:, idx], y_pred_test[:, idx])
                args_to_writer.c_index_train = get_ci(y_train, y_pred_train, args.window_list, idx)
                args_to_writer.c_index_val = get_ci(y_test, y_pred_test, args.window_list, idx)
                args_to_writer.pred_mean = np.mean(y_pred_test[:, idx])
                args_to_writer.n_train = X_train[0].shape 
                args_to_writer.n_val = X_test[0].shape 
                self.write_auc(args_to_writer)
            self.write_history(args_to_writer, train_hist)
        except:
            self.write_error()

def get_ci(y, pred, windows, idx):
    #this y is the output from nnet_survival
    failure = np.sum(y[:, len(windows):], axis=-1)
    failure_idx = np.sum(y[:, :len(windows)], axis=-1)
    failure_idx[failure_idx==6] = 5
    failure_time = np.array([windows[int(f_idx)] for f_idx in failure_idx])
    c_index, _, _, _, _ = ci(failure.astype(bool), failure_time, 1 - pred[:, idx])
    return c_index