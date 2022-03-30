import traceback
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


def update_train_results(X_train, y_train, X_test, y_test, model, filepath, train_hist, df_writer, args):                            
    try:
        #reload best weights
        model.load_weights(filepath)

        # calculate model prediction classes    
        y_pred_train = model.predict(X_train) 
        y_pred_test = model.predict(X_test) 
        
        y_pred_train_binary = (y_pred_train > 0.5).astype('int32')
        y_pred_test_binary = (y_pred_test > 0.5).astype('int32')
        
        args_to_writer = deepcopy(args)

        #add here all the args you want to write in the tables
        args_to_writer.auc_train = roc_auc_score(y_train, y_pred_train)
        args_to_writer.auc_val = roc_auc_score(y_test, y_pred_test)
        args_to_writer.matthews_train = matthews_corrcoef(y_train, y_pred_train_binary)
        args_to_writer.matthews_val = matthews_corrcoef(y_test, y_pred_test_binary)
        #args_to_writer.n_train = y_train.shape[0] 
        #args_to_writer.n_val = y_test.shape[0]
        args_to_writer.n_train = X_train.shape 
        args_to_writer.n_val = X_test.shape 
    
        # add args to 
        args_to_cond_auc = deepcopy(args)
        args_to_cond_auc.pred = y_pred_test
        args_to_cond_auc.truth = y_test

        # calculate conditional AUC and to file
        args_to_writer.auc_cum_val = df_writer.write_conditional_prob(args_to_cond_auc) #thils file compute auc_cum also write in auc temp a DF with the prediction for each patient
        #df_writer.write_auc_cum(args_to_cond_auc)

        # append AUC score to existing file
        df_writer.write_auc(args_to_writer)
        # append other scores to existing file
        df_writer.write_history(args_to_writer, train_hist)
    except:
        df_writer.write_error()
