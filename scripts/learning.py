import traceback
import numpy as np
import pandas as pd
import data_utils, metrics
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Masking
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.utils import class_weight
import writer
import model_generator
import metrics
import pickle
import hashlib

stripper = lambda x : x[0] if ((type(x) is tuple or type(x) is list)and len(x)==1)  else x


def train_model(X_window, y_window , args, df_writer):

    ### MODEL ARCHITECTURE ###

    initial_model_weights_to_encode = '.'.join([str(header) + '_' + str(stripper(args.__dict__[header])) for header in df_writer.AUCheader if header in args.__dict__.keys() and header!= 'cv_num'])
    assert 'cv_num' not in initial_model_weights_to_encode
    initial_model_weights = hashlib.md5(initial_model_weights_to_encode.encode()).hexdigest()

    initial_weights_path = args.output_dir  + "/best_weights/initial_weights.{}.hdf5".format(initial_model_weights)
    pids = X_window[0].pids

    for enum, cv_num in enumerate(args.cv_split.keys()): 

        args.cv_num = cv_num
        name_model_weights_to_encode = '.'.join([str(header) + '_' + str(stripper(args.__dict__[header])) for header in df_writer.AUCheader if header in args.__dict__.keys()])
        name_model_weights = hashlib.md5(name_model_weights_to_encode.encode()).hexdigest()
        args.hash_id = name_model_weights
        model = model_generator.get_model(X_window, args)
        model.summary()
        
        if enum == 0 :
            model.save_weights(initial_weights_path)

        this_train_idx = ~np.isin(np.arange(X_window[0].inputs.shape[0]), args.cv_split[cv_num])
        this_val_idx = np.isin(np.arange(X_window[0].inputs.shape[0]), args.cv_split[cv_num])

        # if df_writer already wrote this config it means that we can skip this experiment
        if df_writer.is_duplicate(args) and not args.force_training:
            continue

        # load the initial weights
        model.load_weights(initial_weights_path) 
        # split in TRAIN and TEST
        this_X_train = [x.inputs[this_train_idx] for x in X_window]
        this_X_val = [x.inputs[this_val_idx] for x in X_window]
        this_y_train, this_y_val = y_window[this_train_idx], y_window[this_val_idx]
        for ii in range(len(args.window_list)):
            assert np.sum(this_y_val[:, ii]) > 0 
            assert np.sum(this_y_train[:, ii]) > 0 
        this_pids_train, this_pids_val = pids[this_train_idx], pids[this_val_idx]

        if type(this_X_train) is list:
            batch_size = this_X_train[0].shape[0] if args.batch == 'all' else int(args.batch)
        else:
            batch_size = this_X_train.shape[0] if args.batch == 'all' else int(args.batch)
        
        # define checkpoint
        filepath = args.output_dir + "/best_weights/{}.hdf5".format(args.hash_id)
        
        checkpoint = ModelCheckpoint(filepath, monitor=args.monitor_checkpoint[0], verbose=args.verbose, save_best_only=True, 
                                                                    mode=args.monitor_checkpoint[1])

        # define early stopping
        earlystopping = EarlyStopping(monitor=args.monitor_early_stopping[0], min_delta=0, patience=args.patience, 
                                                                    verbose=args.verbose, mode=args.monitor_early_stopping[1])
        
        # define callbacks_list
        callbacks_list = []
        if args.early_stopping:
            callbacks_list.append(earlystopping) 
        if args.save_checkpoint:
            callbacks_list.append(checkpoint)

        # TRAIN MODEL
        if args.data == "balanced":
            train_hist = model.fit_generator(data_utils.generate_balanced_arrays(this_X_train, this_y_train, batch_size, len(args.window_list)),
                                        callbacks = callbacks_list,
                                        epochs=args.n_epochs,
                                        validation_data = [this_X_val ,this_y_val],
                                        steps_per_epoch=1,
                                        verbose=args.verbose)
        
        
        elif args.data == "unchanged":
            class_weights = class_weight.compute_class_weight('balanced', np.unique(this_y_train), this_y_train)
            print ("Using class weights .. ", class_weights)
            train_hist = model.fit(this_X_train, this_y_train,
                                        callbacks = callbacks_list,
                                        epochs=args.n_epochs,
                                        validation_data = [this_X_val, this_y_val],
                                        batch_size = batch_size,
                                        class_weight=class_weights,
                                        verbose=args.verbose)

        df_writer.update_train_results(this_X_train, this_y_train, this_X_val, this_y_val, model, filepath, train_hist, df_writer, args)
        
        model_json_path = "{}/best_weights/{}_model.json".format(args.output_dir, args.hash_id)
        
        with open(model_json_path, "w") as json_file:
            json_file.write(model.to_json())
        
        with open("{}/best_weights/{}.x_val.npy".format(args.output_dir, args.hash_id), 'wb') as f:
            pickle.dump(this_X_val, f)
        
        np.save("{}/best_weights/{}.y_val.npy".format(args.output_dir, args.hash_id), this_y_val)
        np.save("{}/best_weights/{}.pids_val.npy".format(args.output_dir, args.hash_id), this_pids_val)


    print ("Training finished")
