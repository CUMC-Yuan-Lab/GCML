import os
if 'PANSEG_PREPROCESSED_DATA' not in os.environ:
    print('Please set PANSEG_PREPROCESSED_DATA first.')
    exit()
else:
    print('PANSEG_PREPROCESSED_DATA:', os.environ['PANSEG_PREPROCESSED_DATA'])
    data_dir = os.environ['PANSEG_PREPROCESSED_DATA']

if 'PANSEG_RAW_DATA' not in os.environ:
    print('Please set PANSEG_RAW_DATA first.')
    exit()
else:
    print('PANSEG_RAW_DATA:', os.environ['PANSEG_RAW_DATA'])
    orig_data_dir = os.environ['PANSEG_RAW_DATA']

param_dir = './outputs_train/'
output_dir = './outputs_predict/'

fname_list = 'case_list_test.csv'

labeled = True

setting={'orig_data_dir':orig_data_dir,
         'data_dir':data_dir,
         'param_dir': param_dir,
         'output_dir':output_dir,
         'fname_list':fname_list,
         'labeled':labeled
}
