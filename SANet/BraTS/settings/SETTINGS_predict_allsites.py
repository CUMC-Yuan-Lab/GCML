# data_dir = '/data/lab/shared_data/BraTS21/BraTS2021_TrainingData/'

import os
if 'INPUT_BRATS_DATA' not in os.environ:
    print('Please set INPUT_BRATS_DATA first.')
    exit()
else:
    print('INPUT_BRATS_DATA:', os.environ['INPUT_BRATS_DATA'])
    data_dir = os.environ['INPUT_BRATS_DATA']

param_dir = './outputs_train/'
output_dir = './outputs_predict/'

fname_list = 'segtraining21-tr-allsites.csv'
labeled = True

setting={
         'data_dir':data_dir,
         'param_dir': param_dir,
         'output_dir':output_dir,
         'fname_list':fname_list,
         'labeled':labeled
}
