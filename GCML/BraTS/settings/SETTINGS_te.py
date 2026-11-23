# Settings for testing (gcml_brats_predict.py --setting SETTINGS_te, the default).
# The data location is shared with the training settings.
from settings.SETTINGS_tr import data_dir

# list of the test cases, one case per line; M and N of gcml_brats_predict.py select the rows to predict
fname_list = 'configs/segtraining21-te-real8.csv'
# the ground truth (<case>_seg.nii.gz) is available, so Dice scores are computed
labeled = True

setting = {'data_dir': data_dir,
           'fname_list': fname_list,
           'labeled': labeled}
