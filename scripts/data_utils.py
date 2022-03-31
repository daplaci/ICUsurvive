import numpy as np
import random
from sklearn.model_selection import StratifiedKFold

stripper = lambda x : x[0] if ((type(x) is tuple or type(x) is list) and len(x)==1)  else x

def init_cv_split(X, y, args):
    cv_split = {}

    kfold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    cv = 0
    
    bins = np.asarray(args.window_list)
    y_binned = np.digitize(y, bins)# double check this

    for train_idx, val_idx in kfold.split(X, y_binned):
        cv += 1
        cv_split[cv] = np.asarray(val_idx)
    return cv_split

def generate_balanced_arrays(X_train, y_train, batch_size, n_intervals):

    while True:
        positive = np.where(y_train[:,(n_intervals-1)]==1)[0].tolist()
        
        if type(batch_size) is int:
            positive = [random.choice(positive) for i in range(batch_size//2)]
        
        negative = np.random.choice(np.where(y_train[:,(n_intervals-1)]==0)[0].tolist(), size = len(positive), replace = False)
        balance = np.concatenate((positive, negative), axis=0)
        
        np.random.shuffle(balance)
        if  isinstance(X_train,np.ndarray):
            inputs = X_train[balance]
        else:
            inputs = stripper([x[balance] for x in X_train])

        target = y_train[balance]
        yield inputs, target
        
