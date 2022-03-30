# import libraries
import sys, os, glob
import pandas as pd
import numpy as np
import datetime
import pytz
from tqdm import tqdm
from scipy import stats
from multiprocessing import Pool
from p_tqdm import *
from functools import partial

output_path = '/home/projects/icu/gum/input/medication_agg_temp/'
metadata_path = '/home/projects/icu/gum/metadata/'
medication_data_path = '/home/projects/icu/gum/input/medication/'

# OBS: Binary!!!

# set parameters
cis_delete_before_t0 = True
max_obs_period_hours = False
add_static_features = True  
testing = True
save_data = True
to_binary = False
to_percentile = True
agg_period_min = 60 
n_cores = 100


#######################################################################
# until now data are selected by courseid's in 'metadata_all.tsv' !!! #
#######################################################################

### METADATA ###
# change directory and load metadata file
os.chdir(metadata_path)
metadata = pd.read_csv('metadata_pre.tsv', sep='\t', parse_dates=['admdatetime_utc'])

# calculate admission in hours
metadata['admission_hours'] = metadata.apply(
        lambda row: row['losmin'] / 60, axis=1
)

metadata_discharge = metadata[['courseid_unique', 'admission_hours']]

# find length of the longest admission in hours
max_admission_hours = int(metadata.losmin.max()/60)

# select columns for 'metadata_admission' data frame
metadata_admission = metadata[['courseid_unique', 'admdatetime_utc']]

# initialize data frame
data_all = pd.DataFrame(columns=['courseid_unique', 'rel_time'])

# include all features with the right name
search_for = medication_data_path + '*.tsv'
fnames = glob.glob(search_for)

def process_file(file):
    
    # extract feature name
    feature = file.split('/')[-1].split('.')[0]

    # read data
    if testing:
        data = pd.read_csv(file, sep='\t', encoding='utf_8_sig', parse_dates=['timestamp'], nrows=1000)
    else:
        data = pd.read_csv(file, sep='\t', encoding='utf_8_sig', parse_dates=['timestamp'])
    
    # fill NA with 0
    #data.fillna(value=0, inplace=True)
    
    # sum "feature columns"
    exclude_cols = ['courseid_unique', 'timestamp', 'epaspresbaseid']
    feature_columns = [col for col in data.columns.tolist() if col not in exclude_cols] 
    data['sum_of_features'] = data[feature_columns].sum(axis=1)
    data.drop(feature_columns, axis=1, inplace=True)
    data.drop(['epaspresbaseid'], axis=1, inplace=True)
     
    # merge with 'metadata_admission' to get 'admdatetime_utc' and select suitable courseid's
    data = pd.merge(data, metadata_admission, on='courseid_unique')
    
    # only proceed if data frame is not empty
    if not data.empty:
    
        # calculate 'rel_time' in minutes
        data['rel_time'] = np.floor(data['timestamp'].sub(data['admdatetime_utc']) / np.timedelta64(agg_period_min, 'm'))
        
        # delete CIS-data before t0
        if cis_delete_before_t0:
            data = data[data.rel_time >= 0]
            
        # delete observation after max_obs_period_hours / max_admission_hours
        if max_obs_period_hours:
            data = data[(data['rel_time'] <= max_obs_period_hours*60/agg_period_min)]
        else:
            data = data[(data['rel_time'] <= max_admission_hours*60/agg_period_min)]
        
        # aggregate
        try:
            data_agg = data.groupby(['rel_time', 'courseid_unique']).agg(['sum']).reset_index()
            data_agg.columns = ['rel_time', 'courseid_unique', feature]
            data_agg = data_agg[['courseid_unique', 'rel_time', feature]]
            
            if to_binary:
                # set values >0 to 1
                mask = data_agg[feature] > 0
                data_agg.loc[mask, feature] = 1
        
            return data_agg
            
            #data_agg[feature]  
            #data_all = pd.merge(data_all, data_agg, on=['courseid_unique', 'rel_time'], how='outer')

        except Exception as e:
            print (e)
            data_agg = 0
            # print(data_all.head())
            # print(data_agg.head())
            sys.exit()



data_agg_to_merge = Pool(n_cores).map(process_file, fnames)


for i, data_agg in enumerate(tqdm(data_agg_to_merge)):
    if data_agg is not None:
        data_all = pd.merge(data_all, data_agg[['courseid_unique', 'rel_time']], on=['courseid_unique', 'rel_time'], how='outer')

def parallel_merging(data_agg, data_all):
    #data_all, data_agg = data_zipped
    if data_agg is not None:
        extended_data_all = pd.merge(data_all, data_agg, on=['courseid_unique', 'rel_time'], how='outer')
        return extended_data_all

data_to_extend = Pool(n_cores).map(partial(parallel_merging,data_all= data_all), data_agg_to_merge)
data_to_extend = [df for df in data_to_extend if df is not None]

for df in tqdm(data_to_extend):

    assert np.array_equal(df.courseid_unique.values,data_all.courseid_unique.values)
    assert np.array_equal(df.rel_time.values, data_all.rel_time.values)
    data_all[df.columns[-1]] = df[[df.columns[-1]]]
    
print ("Dataset created!!")


data_all.fillna(value=0, inplace=True)
data_all = data_all.sort_values(by=['courseid_unique', 'rel_time'])

def add_features(data_all):
    relevant_columns = [x for x in list(data_all.columns) if not x in ['courseid_unique', 'rel_time']]
    mask = data_all.loc[:, relevant_columns] > 0

    # make a list medications
    if to_binary:
        data_all['medication'] = pd.DataFrame(np.where(mask, relevant_columns, ''), columns=relevant_columns).values.tolist()
    else:
        if to_percentile:
            print ("converting to percentile")
            data_all[relevant_columns]= data_all[relevant_columns].replace(0,np.nan)
            data_all[relevant_columns] = data_all[relevant_columns].rank(pct=True, na_option='keep')
            data_all[relevant_columns] = data_all[relevant_columns]*100
            data_all[relevant_columns]= data_all[relevant_columns].replace(np.nan, 0)

        print ("claening data all")
        data_all[relevant_columns] = data_all[relevant_columns].applymap(np.int64)
        feature_str = np.where(mask, relevant_columns, '')
        feature_value = np.where(mask, data_all[relevant_columns], '')
        merged = np.core.defchararray.add(np.core.defchararray.add(feature_str, '_'), feature_value)
        data_all['medication'] = pd.DataFrame(merged, columns=relevant_columns).values.tolist()
    def remove_empty(oldlist):
        return [x for x in oldlist if x not in ['', '_']]

    print ("removing empty ")
    data_all['medication'] = data_all.medication.apply(remove_empty) 
    
    return data_all[['courseid_unique', 'rel_time', 'medication']]

def parallelize_dataframe(df, func, n_cores=n_cores):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

data_all_parallel = parallelize_dataframe(data_all, add_features)


if save_data:
    print ("saving  ")
    # change directory and save data file
    file_name = output_path + str(agg_period_min) + 'min_medication.tsv'
    data_all_parallel.to_csv(file_name, sep='\t', header=True, index=False, encoding='utf_8_sig')
