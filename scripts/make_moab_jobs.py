#!/usr/bin/python

import os, itertools, time, shutil, sys, stat, subprocess, pdb
import utils

#here the additional parameter that are not included in the grid search are specified
args = utils.parse_args()
moab_params = [k for argv in sys.argv[1:] for k in args.__dict__.keys() if k in argv]
additional_params = ' '.join([' --'+str(k)+ '='+ str(args.__dict__[k]) for k in moab_params])

print ("additional params:", additional_params)
# check if Python-version > 3.0
# with Python2 the dict is not in alphabetic order
assert (sys.version_info > (3, 0)), "This script only works with Python3!"

script_dir = '/home/people/daplaci/git_gum/scripts/'
base_dir = '/home/people/daplaci/git_gum/'
input_dir = '/home/people/daplaci/git_gum/input/'
output_dir = '/home/people/daplaci/git_gum/output/'
nnet_file = 'hyperopt_search.py'
batch_file = 'run_models.sh'
list_file_to_copy = os.listdir(script_dir)


# make parameter dictionary
param_dict={}
param_dict['n_window'] = ['1-7-14-30-90-365']
# param_dict['recurrent_layer'] = ['lstm']
param_dict['baseline_hour'] = [0,24,48,72]
# param_dict['units'] = [216]
# param_dict['batch'] = [128]
# param_dict['l2_regularizer'] = [0.000973591634224646]
# param_dict['embedding_coeff'] = [0.136160010477897]
# param_dict['dropout'] = [0.143712093485934]
# param_dict['recurrent_dropout'] = [0.885876590955943]
# param_dict['padd_percentile'] = [94]
# param_dict['optimizer'] = ["Adagrad"]
# param_dict['save_all'] = [True]
# param_dict['exp_id'] = ["4ecb89aac9844a3b862728d77879f1a4"]


# do not run this, if an other script is just trying to import 'param_dict'
if __name__ == '__main__':

    unknown_param = [k for k in list(param_dict.keys()) if k not in args.__dict__.keys()]
    if any(unknown_param): 
        raise Warning("Warning: these parameters of the grid search are not initialized in the main code:")
        print ('\n'.join(unknown_param))

    # make list of all argument combinations
    all_args = (dict(zip(param_dict, x)) for x in itertools.product(*param_dict.values()))
    
    for arg in all_args:
    
        # change directory and open files for writing

        date_string = time.strftime("%Y-%m-%d-%H%M")
        name_folder_add = input("""Type the name of the output folder to append at the current date time for the args {}:
Insert a string without spaces: """.format(arg))
        date_string += name_folder_add
        
        os.chdir(output_dir)
        
        os.mkdir(date_string)
        path = "".join([output_dir, date_string])
        if args.debug: print ("Run your grid_search in the folder called ", path)
        os.chdir(path)
        os.mkdir('best_weights')
        os.mkdir('auc_temp')
        os.mkdir('moab_jobs')
        os.mkdir('data')
        os.mkdir('logs')
        os.system('chmod -R 700 moab_jobs/')
        
        # save this script to path
        filename = "".join([script_dir, sys.argv[0]])
        shutil.copy2(filename, path)
        for file_to_copy in list_file_to_copy:
            if ('.py' in file_to_copy) or ('.R' in file_to_copy):
                shutil.copy2(os.path.join(script_dir, file_to_copy), path)
                
        infile = open('eval_gridsearch.R', 'r') 
        rscript = infile.read()
        rscript = rscript.replace('#placeholder#', output_dir + date_string)
        infile.close()
        outfile = open('eval_gridsearch.R', 'w')
        outfile.write(rscript)
        outfile.close()
          
        # make batch-file
        batch_file = open('run_models.sh', 'w')
        batch_file.write('#!/bin/bash\n')
        batch_file.write('cd ' + output_dir + date_string + '/\n')
        batch_file.write('for f in moab_jobs/*.sh\n')
        batch_file.write('do\n')
        batch_file.write('\tqsub $f\n')
        batch_file.write('done\n')
        batch_file.close()
        # change permissions
        st = os.stat('run_models.sh')
        os.chmod('run_models.sh', st.st_mode | stat.S_IEXEC)
        
        # create output files
        AUCfile = open('AUC_history_gridsearch.tsv', 'w')
        CVfile = open('CV_history_gridsearch.tsv', 'w')

        
        AUCheader = ['auc_train', 'auc_val', 'c_index_train', 'c_index_val', 'pred_mean', 'cv_num', 'n_train', 'n_val', 'hash_id', "exp_id"]
        [AUCheader.append(k) for k in list(param_dict.keys()) if k not in AUCheader]
        CVheader = ['acc', 'loss',  'matthews', 'precision', 'recall',  'val_acc', 'val_loss', 'val_matthews', 
                            'val_precision',  'val_recall', 'pr ed_mean', 'cv_num', 'hash_id', "exp_id"]
        [CVheader.append(k) for k in list(param_dict.keys()) if k not in CVheader]
        #here a new file should be added to plot the cumulative AUC
        AUCcum_header = ['auc_val', 'cv_num']  + list(param_dict.keys())
                            
        CVfile.write('epoch\t' + '\t'.join(CVheader) + '\n')
        AUCfile.write('\t'.join(AUCheader) + '\n')

        # close files
        AUCfile.close()
        CVfile.close()
        
        # open log files
        error_log = open('error.log', 'w')
        errorHeader = ['error', 'args']
        error_log.write('\t'.join(errorHeader) + '\n')
        error_log.close()
        
        progress_log = open('progress.log', 'w')
        progressHeader = ['completed']
        progress_log.write('\t'.join(progressHeader) + '\n')
        progress_log.close()
        
        
        CMD = '''#!/bin/bash
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=100gb
#PBS -l walltime=10:00:00:00
#PBS -e logs/$PBS_JOBID.err
#PBS -o logs/$PBS_JOBID.log
#PBS -N icu
'''
        if not args.debug:

            shell_arg = 'python -u ' + nnet_file + ' --date_string=' + date_string + additional_params
            file_name =  []
            for key in arg:
                shell_arg += ' --' + key + '=' + str(arg[key])
                file_name += [str(key) + '_' + str(arg[key])]
            if "exp_id" in arg:
                file_name_joined = '.'.join([f for f in file_name if 'exp_id' in f])
            else:
                file_name_joined = '.'.join([param for header in AUCheader for param in file_name if header in param])

            shell_arg += " > logs/{}.log 2>&1".format(file_name_joined) ####check this __it can be wrong 
            shellfile = open('moab_jobs/%s.sh' % file_name_joined, 'w')
            shellfile.write(CMD)
            shellfile.write('cd ' + path + '\n')
            shellfile.write(shell_arg + '\n')#+ ' >logs/{}.log 2>&1 &\n'.format(file_name_joined)
            shellfile.close()
        
            #run batch-scripts
            run_models_path = '/'.join([path, r'run_models.sh'])
            subprocess.call([run_models_path])
          
          #subprocess.call('nohup python -u moab_watcher.py > moab_watcher.log 2>&1 &', shell=True) # -u streams the output directly to the output file
          
