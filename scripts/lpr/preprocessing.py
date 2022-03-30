import ast
from itertools import groupby
import numpy as np


def get_code(metadata, args):
    def trunc_code(code):
        if code[0] == 'D' and args.truncation:
            return code[:args.truncation +1]
        elif code.isdigit() and args.truncation:
            return code[:args.truncation]
        elif (code[0] == 'Y' or code[0] == 'E') and args.truncation:
            return code[:args.truncation +1]
        else:
            return code
    
    metadata[args.lpr_code_type] = metadata[args.lpr_code_type].apply(lambda d: ast.literal_eval(d))
    metadata[args.lpr_code_type + '_timedelta_h'] = metadata[args.lpr_code_type + '_timedelta_h'].apply(lambda d: ast.literal_eval(d))
    metadata[args.lpr_code_type] = metadata[args.lpr_code_type].apply(lambda d: list(map(trunc_code, d)))
    
    metadata['zipped'] = 0
    metadata['zipped'] = metadata.apply(lambda x: list(zip(x[args.lpr_code_type],x[args.lpr_code_type + '_timedelta_h'])), axis=1)

    metadata['zipped'] = metadata['zipped'].apply(lambda d : [list(g)[0] for k,g in groupby(d, lambda x :x [0])])
    
    #
    metadata[args.lpr_code_type] = list(map(lambda x: list(zip(*x))[0], metadata.zipped.tolist()))
    metadata[args.lpr_code_type + '_timedelta_h'] = list(map(lambda x: list(zip(*x))[1], metadata.zipped.tolist()))
    
    #remove the zeros and apply log to time delta
    metadata[args.lpr_code_type + '_timedelta_h'] = metadata[args.lpr_code_type + '_timedelta_h'].apply(lambda x: list(map(lambda y: y if y!=0 else 1, x)))
    metadata[args.lpr_code_type + '_timedelta_h'] = metadata[args.lpr_code_type + '_timedelta_h'].apply(lambda x: np.log(x))
    
    #convert to list
    metadata[args.lpr_code_type] = metadata[args.lpr_code_type].apply(list)
    metadata[args.lpr_code_type + '_timedelta_h'] = metadata[args.lpr_code_type + '_timedelta_h'].apply(list)
    
    #padd the time delta
    metadata[args.lpr_code_type + '_timedelta_h'] = metadata[args.lpr_code_type + '_timedelta_h'].apply(lambda x: [x[0]]* (args.len_seq - len(x)) + x[-args.len_seq:])

    #convert to str
    metadata[args.lpr_code_type] = metadata[args.lpr_code_type].apply(str)

    return metadata

def remove_nan(metadata, args):
    metadata = metadata.dropna()
    metadata = metadata.reset_index(drop=True)
    #metadata[args.lpr_code_type] = metadata[args.lpr_code_type].apply(lambda d: d if isinstance(d, str) else '[]')
    #metadata[args.lpr_code_type + '_timedelta_h'] = metadata[args.lpr_code_type + '_timedelta_h'].apply(lambda d: d if isinstance(d, str) else '[]')
    return metadata