config = {'tag': None,
          'prefix': 'FL',
          'prep_ver':1, 
          'NCV': 5,
          'margin': 8, # expand from 8 for testing only (docker submission), # margin in x, y and z directions, voxels
          'imgsp':(1.0, 1.0, 1.0), # not used for localization
          'img_szx': 240,
          'img_szy': 240,
          'img_szz': 155,
          'crop_szx':128,
          'crop_szy':128,
          'crop_szz':128,
          'OARs': {'WT':1, 'TC':2, 'ET': 3}, # WT: whole tumor; TC: tumor core; ET: enhanced tumor
          'img_ch': ['t1','t1ce','t2','flair'], # number of image channels
          'lower_bound': None,
          'upper_bound': None, # Slicer default w/l for abdomen
          'z_transform': None, #[0.4688,0.1938],
          'hq': False, # if voi_histogram_equalization will be applied
          'shift': None,#20,
          'rotate': 0,#15,#20,
          'scale': 0,#0.15,#0.1,#0.15,
          'p_lr': 0.5,
          'p_ud': 0.5,
          'p_si': 0.5,
          'p_rot90':0,
          'pct_norm_tr': None,#((1,20), (90,100)),
          'pct_norm': None,#(10, 95),
          'contrast': (0.9,1.1),#None, # (0.75,1.25),
          'gamma': None, # (0.5,2.0),
          'augument_fcn':'img_augument3d',
          'model': 'sanet1_2_0', #'fcresunet_ss2', # scale-attention network
          'up_mode': 'upconv',
          'group_norm': 1,
          'dropout_p' : 0.2,
          'depth': 5,
          'nf': 12, #24,
          'optim': 'adam',
          'loss':'bja_bfl_orl', # BinaryJaccardLoss + BinaryFocalLoss
          'val_loss':'bja_orl', #BinaryJaccardLoss 
          'weight_sample':[0.17, 0.33, 0.50],#None,
          'dp':True, # deep supervision
          'val_ma_alpha':0.9, # moving average of validation loss
          'grad_norm_clip':0.1, # gradient clip
          'pretrained': None,
          'gan':None,
          'tta':False, # test-time augumentation
          'n_warmup': 0,
          'fname_list':'segtraining21-tr-pooled.csv', #place holder
          'index_train':[1,87], #place holder
          'index_valid':[88,102], #place holder
          'index_test':[0,0], #place holder
          'nodes': ['4', '5', '6', '7', '11', '16', '20', '21']  # real8
          }

