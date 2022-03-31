import argparse
import os
from sklearn.metrics import roc_auc_score


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
    # What to execute
    #training params
    parser.add_argument('--n_epochs', type=int, default=8000, help='number of epochs for train [default: 256]')
    parser.add_argument('--patience', type=int, default=500, help='epochs to wait before stopping')
    parser.add_argument('--recurrent_dropout', type=float, default=0.4, help='recurrent dropuout to apply')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropuout to apply')
    parser.add_argument('--layers', type=int, default=1, help='number of layers to use')
    parser.add_argument('--units', type=int, default=64, help='number of units to use')
    parser.add_argument('--embedding_dim', type=int, default=256, help='seed for random')
    parser.add_argument('--embedding_coeff', type=float, default=2, help='coefficient used for computing the emb size from the vocab size')
    parser.add_argument('--l2_regularizer', type=float, default=1e-4, help='l2 regularizer to apply in the model')
    parser.add_argument('--truncation', type=int, help='if an integer is specified, is how we want to truncate the ')
    parser.add_argument('--num_heads', type=int, default=4, help='if an integer is specified, is how we want to truncate the ')
    parser.add_argument('--size_head', type=int, default=32, help='if an integer is specified, is how we want to truncate the ')

    parser.add_argument('--test_size', type=float, default=0.2, help='percentage of test to held out')
    parser.add_argument('--activation', type=str, default='sigmoid', help='activation for net')
    parser.add_argument('--seed', type=int, default=7, help='seed for random')
    parser.add_argument('--batch', type=str, default='128', help='batch_size')
    parser.add_argument('--masking', type=str2bool, default=True, help='bool for masking')
    parser.add_argument('--early_stopping', type=str2bool, default=True, help='wheter to apply early stopping')
    parser.add_argument('--save_checkpoint', type=str2bool, default=True, help='wheter to save checkpoint')
    parser.add_argument('--optimizer', type=str, default='Adagrad', help='optmizer for masking')
    parser.add_argument('--exp_id', type=str, help='id for the experiment')

    parser.add_argument('--date_string', type=str, help='datestring')
    parser.add_argument('--data', default='balanced', type=str, help='data how to balance the data')
    parser.add_argument('--recurrent_layer', default='gru', type=str, help='which kind od recurrent')
    parser.add_argument('--bidirectional', default=False, type=str2bool, help='if bidirectional')
    parser.add_argument('--use_gpu', default=True, type=str2bool, help='wheter to use gpu')

    parser.add_argument('--n_window', default='1-30-90', type=str, help='prediction window #')    
    parser.add_argument('--baseline_hour', default=24, type=int, help='time for prediction (hours since adm)' )
    parser.add_argument('--window_hours', default=24,  type=int, help='prediction window (hours)' )
    parser.add_argument('--explain', type=str2bool, default=False, help='if true run the explanation')

    parser.add_argument('--n_splits', type=int, default=5, help='number of splits for cross validation')
    parser.add_argument('--verbose', type=int, default=1, help='training verbose')

    parser.add_argument('--dataset_nme', type=str, default='train.npy', help='npy file for loading the data') #alternative -- note
    parser.add_argument('--compartment', type=str, default='ehr', help='compartment to train the data on')
    
    parser.add_argument('--force_training', type=str2bool, default=False, help='if true the experiments will be overwritten')
    parser.add_argument('--debug', type=str2bool, default=False, help='run scripts in debug-mode')
    parser.add_argument('--save_all', type=str2bool, default=False, help='if True, all the data will be saved')
    parser.add_argument('--max_num_patients', type=int, help='max number of patients to include')
    # default args
    parser.add_argument('-f', '--file', type=str, default='filepath', help='flag for jupyter')
    parser.add_argument('--diag', type=str2bool, default=True, help='include diag code')
    parser.add_argument('--opr', type=str2bool, default=True, help='include opr code')
    parser.add_argument('--ube', type=str2bool, default=True, help='include ube code')
    parser.add_argument('--chart', type=str2bool, default=True, help='include chart code')
    parser.add_argument('--medication', type=str2bool, default=True, help='include medication code')
    parser.add_argument('--equipment', type=str2bool, default=True, help='include equipment code')
    parser.add_argument('--biochem', type=str2bool, default=True, help='include biochemical code')
    parser.add_argument('--ageadm', type=str2bool, default=True, help='include equipment code')
    parser.add_argument('--los_before_icu', type=str2bool, default=True, help='include biochemical code')
    parser.add_argument('--padd_percentile', type=int, default=95, help='number of padd diag code')
    parser.add_argument('--min_word_count', type=int, default=100, help='minimum number of words to be added in the dictionary')
    parser.add_argument('--diag_level', type=str, default="block", help='decide between block and l3')
    
    #args for notes
    args = parser.parse_args()
  
    if '-' in args.compartment:
        args.compartment = args.compartment.split('-')
    else:
        args.compartment = (args.compartment, )
    
    #parse winndows as a list
    args.window_list = [int(window) for window in args.n_window.split('-')] #[1,30,90]

    # checkpoint
    args.monitor_checkpoint = ['val_loss', 'min']
    # early_stopping
    args.monitor_early_stopping = ['val_loss', 'min']


    args.base_dir = '/home/people/daplaci/git_gum/'
    args.script_dir = '/home/people/daplaci/git_gum/scripts/'
    args.input_dir = '/home/people/daplaci/git_gum/input/'
    if args.date_string:
        args.output_dir = '/home/people/daplaci/git_gum/output/' + args.date_string
    else:
        args.output_dir = os.getcwd()
    args.lpr_metadata = args.input_dir + 'list_of_codes_before_adm.tsv'
    args.metadata_path = args.base_dir + 'metadata/metadata.tsv'
    args.notes_metadata_path = args.base_dir + 'metadata/metadata_notes.tsv'

    args = apply_restrictions(args)

    return args

def apply_restrictions(args):
    if args.apply_post_attention and args.apply_pre_attention and args.batch != 'all' and int(args.batch)>32:
        raise Warning("batch size might be too high if both the attention are used")

    if ('lpr' in args.compartment) and  args.batch=='all':
        raise Warning("You are using LPR data for training with the maximum batch size.. it will cause memori exception")
        
    return args