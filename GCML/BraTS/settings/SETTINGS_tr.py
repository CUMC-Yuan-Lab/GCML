# Settings for training (gcml_brats_train.py --setting SETTINGS_tr, the default).
# data_dir: folder holding one sub-folder per BraTS 2021 case (BraTS2021_00000/, BraTS2021_00002/, ...), each with
# <case>_t1.nii.gz, <case>_t1ce.nii.gz, <case>_t2.nii.gz, <case>_flair.nii.gz and <case>_seg.nii.gz.
# Edit it to point to your copy of the data. Model and log locations are set by --output_dir of the scripts.
data_dir = '/mnt/data/brats2021/BraTS2021_Training_Data_full/'

setting = {'data_dir': data_dir}
