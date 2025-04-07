import argparse, logging
import os, sys, time
import numpy as np
import SimpleITK as sitk

from importlib import import_module
import torch
import torch.nn.functional as F


from torch.nn import DataParallel
from torch.utils.data import DataLoader

# from data.data import DP3DImageDataset

sys.path.append('../')
from utils import setgpu, count_parameters, set_logger, pt_pre_process, overlap_similarities, voi_histogram_equalization

from functools import partial
from SimpleITK import GetArrayViewFromImage as ArrayView

import pandas as pd
from torch.utils.data import Dataset

# from pre_process import pt_pre_process
# from image_processing import voi_histogram_equalization

class DP3DImageDataset(Dataset):
    def __init__(self, config, setting, M, N, transform=None):

        super(DP3DImageDataset, self).__init__()

        self.c = config
        self.s = setting
        full_list = self.s['fname_list']
        skip_rows = M - 1
        self.n_rows = N - M + 1
        df = pd.DataFrame(pd.read_csv(full_list, header=None, skiprows=skip_rows, nrows=self.n_rows))
        self.ptnames = df.values
        self.transform = transform

    def __getitem__(self, item):

        pt = self.ptnames[item][0]
        print('pt = {}'.format(pt))

        if self.c['prep_ver'] in [1,3] :
            image = pt_pre_process(pt, s=self.s, c=self.c, model='test')
            image = np.squeeze(image)
            image = np.rollaxis(image, 3, 0).astype(np.float32)  # [c,z,y,x]


            image = np.swapaxes(image, 1, 3)  # [c,x,y,z]

            if self.c['hq']:
                t1ce = image[1]
                br_mask = np.zeros_like(t1ce)
                br_mask[np.where(t1ce > 0)] = 1
                hq_t1ce = np.expand_dims(voi_histogram_equalization(t1ce, br_mask, number_bins=256), axis=0).astype(
                    np.float32)
                image = np.vstack((image, hq_t1ce))

            if self.transform:
                image = self.transform(image)


        brain_mask = np.zeros(image.shape[1:], dtype=np.uint8)
        for i in range(image.shape[0]):
            brn_idx = np.where(image[i] > 0)
            idx0 = np.where(image[i] == 0)
            brain_mask[brn_idx] = 1
            if self.c['augument_fcn'] in ['img_augument3d']:
                image[i] = (image[i] - np.mean(image[i])) / np.std(image[i])
            else:
                raise ValueError(self.c['augument_fcn'] + ' is not implemented yet!')
        idx = np.where(brain_mask > 0)
        mrgn = self.c['margin']
        cz = [self.c['crop_szx'], self.c['crop_szy'], self.c['crop_szz']]
        voi = []
        for i in range(3):
            brz = max(idx[i].max() - idx[i].min() + 1, cz[i]) + 2 * mrgn
            if brz >= image.shape[i+1]:
                brz = image.shape[i+1]
                b0 = 0
                b1 = brz - cz[i]
            else:
                b0 = max(0, idx[i].min() - mrgn)
                b1 = min(image.shape[i+1], b0 + brz) - cz[i]
                if b1 < b0:
                    b0 = max(0, b1 - 2 * mrgn)
                    brz= b1 - b0 + cz[i]
                if b0 + brz > image.shape[i+1]:
                    b0 = image.shape[i+1] - brz

            voi.append(b0)
            voi.append(brz)

        voi = np.asarray(voi).astype(np.uint8)
        return (image, voi, pt)

    def __len__(self):
        return self.n_rows

    def len(self):
        return self.n_rows


def get_net_and_checkpoints(c, model_loc, ncv=1, mode='best_val_loss'):
    model_name = c['model']
    in_channels = len(c['img_ch'])
    if c['hq']:
        in_channels += 1

    dropout_p = c['dropout_p']
    up_mode = c['up_mode']
    depth = c['depth']
    nf = c['nf']
    gn = c['group_norm']
    nonlin='relu'
    if 'nonlin' in c.keys():
        nonlin = c['nonlin']
    # import model here
    print('loading model: ' + model_name)
    print(' - depth = {}'.format(depth))
    print(' - in_channels = {}'.format(in_channels))
    print(' - nf = {}'.format(nf))
    print(' - dropout_p = {}'.format(dropout_p))
    print(' - up_mode = ' + up_mode)
    print(' - group_norm = {}'.format(gn))

    model = import_module('models.' + model_name)
    n_gpu = setgpu(c['args'].gpu)
    c['args'].n_gpu = n_gpu

    if 'orl' not in c.keys():
        c['orl'] = ('orl' in c['loss'])

    n_classes = 1 if len(c['OARs']) == 1 else len(c['OARs']) + 1

    if c['orl'] and n_classes > 1:
        out_channels = n_classes - 1
    else:
        out_channels = n_classes
    if 'dp' not in c.keys():
        c['dp'] = False
    net = model.Net(in_channels=in_channels, nclasses=out_channels, nf=nf, relu=nonlin, up_mode=up_mode, dropout_p=dropout_p,
                        group_norm=gn, depth=depth, padding=True, deep_supervision=c['dp'])

    n_params = count_parameters(net)
    print(' - param_count = {0:,}'.format(n_params))
    net = net.cuda()
    net = DataParallel(net)

    checkpoints_list = []

    if ncv == 1:
        check_point = torch.load(model_loc)
        checkpoints_list.append(check_point['state_dict'])
        print('loaded single {} model from {}'.format(model_name, model_loc))
    else:
        model_files = filter(lambda s: mode in s, os.listdir(model_loc))
        for pfn in model_files:
            check_point = torch.load(os.path.join(model_loc, pfn))
            checkpoints_list.append(check_point['state_dict'])
            print('loaded {} model from {}'.format(model_name, pfn))

        print('loaded {} {} models in ensemble'.format(len(checkpoints_list), model_name)) #python 3
    return net, checkpoints_list

def test(data_loader, net, ckpt_list, s, c, logger):
    '''
    From BraTS 2020, removed all the ad-hoc post-processing steps
    :param data_loader:
    :param net:
    :param ckpt_list:
    :param s:
    :param c:
    :param logger:
    :return:
    '''
    t0 = time.time()
    net.eval()
    metrics = []

    metrics_str = ''

    cz = np.asarray([c['crop_szx'], c['crop_szy'], c['crop_szz']])
    nch = len(c['img_ch'])
    ncls = len(c['OARs'])
    iz = np.asarray([c['img_szx'], c['img_szy'], c['img_szz']])
    for i, (data, voi, pt) in enumerate(data_loader):
        # print('data.type = {}, size = {}'.format(type(data), data.size()))
        data = torch.squeeze(data)
        voi = torch.squeeze(voi)
        pt = pt[0]
        print('data.size = {}'.format(data.size()))
        print('voi = {}'.format(voi))
        # assert(0)

        xyz = np.zeros((3, 2))
        for j in range(3):
            xyz[j][0] = voi[2 * j]
            xyz[j][1] = voi[2 * j] + voi[2 * j + 1] - cz[j]
        print('xyz = {}'.format(xyz))
        xyz = np.array(np.meshgrid(xyz[0], xyz[1], xyz[2])).T.reshape(-1, 3).astype(np.uint8)

        n_xyz = xyz.shape[0]

        print('n_xyz = {}'.format(n_xyz))
        # assert(0)

        patch_outputs = []
        for check_point in ckpt_list:
            net.load_state_dict(check_point)
            output = np.zeros((n_xyz, ncls, voi[1], voi[3], voi[5])).astype(np.float32)
            output.fill(np.nan)
            for k, (x, y, z) in enumerate(xyz):
                data_k = torch.unsqueeze(data[:, x:x + cz[0], y:y + cz[1], z:z + cz[2]], dim=0)
                print('patch {} '.format(k), (x, y, z))
                with torch.no_grad():
                    data_k = data_k.cuda()
                output_k = net(data_k)
                if isinstance(output_k, list): output_k = output_k[0]
                if c['prob_output'] is False:
                    if c['orl']:
                        output_k = F.sigmoid(output_k)
                    else:
                        output_k = F.softmax(output_k, dim=1)
                output_k = output_k.data.cpu().numpy().squeeze()
                # print(type(output_k), output_k.shape)
                x = x - voi[0]
                y = y - voi[2]
                z = z - voi[4]

                # convert x y z from tensor to int
                x = int(x)
                y = int(y)
                z = int(z)

                output[k, :, x:x + cz[0], y:y + cz[1], z:z + cz[2]] = output_k
            output = np.nanmean(output, axis=0)
            print(output.dtype, output.shape)
            patch_full_output = np.zeros((ncls, iz[0], iz[1], iz[2]), dtype=np.float32)
            patch_full_output[:, voi[0]:voi[0] + voi[1], voi[2]:voi[2] + voi[3], voi[4]:voi[4] + voi[5]] = output
            print('patch_full_output.shape = {}, range = [{}, {}]'.format(patch_full_output.shape,
                                                                          patch_full_output.min(),
                                                                          patch_full_output.max()))

            patch_outputs.append(patch_full_output)
        full_outputs = np.asarray(patch_outputs).astype(np.float32)
        print('full_outputs', full_outputs.shape)
        full_outputs = np.mean(full_outputs, axis=0)

        print('full_outputs.shape = {}, range = [{}, {}]'.format(full_outputs.shape, full_outputs.min(),
                                                                 full_outputs.max()))
        # transfer back to the original dim
        full_outputs = np.swapaxes(full_outputs, 1, 3)  # [c,z,y,x]
        if c['orl']:
            threshold = c['threshold']
            full_mask = np.zeros_like(full_outputs).astype(np.uint8)
            full_mask[np.where(full_outputs >= threshold)] = 1
            full_mask2 = np.zeros_like(full_mask).astype(np.uint8)
            for cls in range(ncls):
                temp_cls = np.sum(full_mask[cls:ncls, :], axis=0)
                temp_cls[np.where(temp_cls > 0)] = 1
                full_mask2[cls] = temp_cls
            full_mask = np.sum(full_mask2, axis=0)
            # full_mask = np.sum(full_mask, axis=0)
        else:
            full_mask = np.argmax(full_outputs, axis=0)
        print('full_mask: shape = {}, valules = {}'.format(full_mask.shape, np.unique(full_mask)))

        # map back to the original labels
        relabled_mask = np.zeros_like(full_mask).astype(np.uint8)
        relabled_mask[np.where(full_mask == 3)] = 4  # enhanced tumor
        relabled_mask[np.where(full_mask == 2)] = 1  # necrotic & non-enhancing
        relabled_mask[np.where(full_mask == 1)] = 2  # peritumoral edema

        # save mask file
        if s['labeled']:
            ref_image_file = os.path.join(s['data_dir'] + '/' + pt, pt + '_seg.nii.gz')
        else:
            ref_image_file = os.path.join(s['data_dir'] + '/' + pt, pt + '_' + c['img_ch'][0] + '.nii.gz')
        ref_img = sitk.ReadImage(ref_image_file)
        mask_img = sitk.GetImageFromArray(relabled_mask)
        mask_img.CopyInformation(ref_img)

        mask_img_file = os.path.join(c['save_dir'], pt + '.nii.gz')  # used to be '_pred.nii.gz' in cross-validation
        sitk.WriteImage(mask_img, mask_img_file)
        print(mask_img_file + ' is saved')

        if s['labeled']:
            full_mask_img = sitk.GetImageFromArray(full_mask)
            full_mask_img.CopyInformation(ref_img)
            seg_data = sitk.GetArrayFromImage(ref_img)
            label_data = np.zeros_like(seg_data).astype(np.uint8)
            label_data[np.where(seg_data > 0)] = 1
            label_data[np.where(seg_data == 1)] = 2
            label_data[np.where(seg_data == 4)] = 3

            label_img = sitk.GetImageFromArray(label_data)
            label_img.CopyInformation(ref_img)
            overlaps = []
            metrics_str += pt

            dice_th = [0.8, 0.5, 0.3]
            warning_head_exist = False
            for OAR, label in sorted(c['OARs'].items(), key=lambda x: x[1]):

                label_mask = sitk.BinaryThreshold(label_img, lowerThreshold=label,
                                                  # upperThreshold=label,
                                                  insideValue=1,
                                                  outsideValue=0)
                pred_mask = sitk.BinaryThreshold(full_mask_img, lowerThreshold=label,
                                                 # upperThreshold=label,
                                                 insideValue=1,
                                                 outsideValue=0)
                overlap = overlap_similarities(label_mask, pred_mask, label=1)
                if overlap['dice'] == float('inf'):
                    dice = np.nan  # or 1.0
                else:
                    dice = overlap['dice']
                print(OAR + ' [{}]: {:5.4f}'.format(label, dice))
                overlaps.append(dice)
                metrics_str += ',{:5.4f}'.format(dice)
                # if dice == 0:
                if dice < dice_th[label - 1]:
                    if warning_head_exist is False:
                        logger.info('[{}] {}'.format(i, pt))
                        warning_head_exist = True
                    n_truth = np.count_nonzero(sitk.GetArrayFromImage(label_mask))
                    n_pred = np.count_nonzero(sitk.GetArrayFromImage(pred_mask))
                    logger.info(
                        '  - [{}], dice = {:5.4}, n_truth = {}, n_pred = {}'.format(OAR, dice, n_truth, n_pred))

            metrics.append([overlaps[i_oar] for i_oar in range(len(c['OARs']))])
            metrics_str += '\n'
    metrics = np.asarray(metrics)

    elpased_time = time.time() - t0
    print('predict {} cases took {:3.2f} s'.format(i + 1, elpased_time))

    return metrics, metrics_str

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch 2D Deep Planning: Dose prediction')
    parser.add_argument('-f', '--fold', type=int, metavar='F',
                        help='the models from cross-validation training')
    parser.add_argument('--gpu', default='0', type=str, metavar='N',
                        help='the GPUs used (default: all)')
    parser.add_argument('-m', '--model', default='best_val_loss',
                        choices=['best_val_loss', 'best_val_loss_ma', 'history'],
                        help='the model used for prediction,, chosen from [best_val_loss, best_val_loss_ma, history]')
    parser.add_argument('--config', '-c', default='config_v0', type=str,
                        help='configuration file for model training/testing')
    parser.add_argument('--setting', default='SETTINGS_predict_allsites', type=str,
                        help='the file specifying where data are read/saved (default: SETTINGS_predict_allsites), placed in settings/.')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    total_time_start = time.time()

    tag = args.config[8:]
    config_training = import_module('configs.' + args.config)
    config = config_training.config
    config['tag'] = tag
    config['threshold'] = 0.5
    config['save_output'] = True
    config['args'] = args

    hostpath = './log/'

    if not os.path.exists(hostpath):
        os.makedirs(hostpath)

    sys.path.append(hostpath)
    # import SETTINGS_te as s
    s = import_module('settings.' + args.setting)
    s = s.setting

    fold = args.fold
    fold_n = 250

    if args.setting == 'SETTINGS_predict_8sites':
        M = 1 # first pt idx
        N = 48 # last pt idx
    elif args.setting == 'SETTINGS_predict_allsites':
        if fold == 0:
            # te_ptnames = ptnames[0:fold_n]
            M = 1
            N = 250
        elif fold > 0 and fold < 4:
            # te_ptnames = ptnames[f*fold_n:(f+1)*fold_n]
            M = fold * fold_n + 1
            N = (fold + 1) * fold_n
        elif fold == 4:
            # te_ptnames = ptnames[f*fold_n:]
            M = fold * fold_n + 1
            N = 1251
        else:
            # te_ptnames = ptnames
            M = 1
            N = 1251

    model_dir = os.path.join(s['param_dir'], args.config)

    if fold is None:
        net, checkpoints_list = get_net_and_checkpoints(config, model_dir, ncv=config['NCV'], mode=args.model)
    else:
        model_fname = args.model + '_f{}'.format(fold)
        model_fname += '.ckpt'
        fcn_file = os.path.join(model_dir, model_fname)
        net, checkpoints_list = get_net_and_checkpoints(config, fcn_file, ncv=1)

    config['save_dir'] = args.config + '/' + args.model
    if args.fold is not None:
        config['save_dir'] = config['save_dir'] + '-f{}'.format(fold)
    config['save_dir'] = os.path.join(s['output_dir'], config['save_dir'])

    print('predicted mask will be saved in :' + config['save_dir'])

    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])
    te_transform = None
    test_dataset = DP3DImageDataset(config, s, M, N, transform=te_transform)
    n_test = test_dataset.len()
    print('n_test = {}'.format(n_test))

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=8, # number of workers
                             pin_memory=True)

    outcsv_file = os.path.join(config['save_dir'],
                               'dice_tag{}_t{}_f{}.csv'.format(config['tag'], config['threshold'], fold))
    log_file = hostpath + config['prefix'] + "_te_v{}_f{}.txt".format(config['tag'], fold)
    if os.path.isfile(outcsv_file): os.remove(outcsv_file)

    # add logger for cv only
    if 'prob_output' not in config.keys():
        config['prob_output'] = False

    if 'tta' not in config.keys():
        config['tta'] = False
    print('tta = {}'.format(config['tta']))

    logger = logging.getLogger(config['prefix'])

    logger = set_logger(logger, log_file, log=False)
    logger.info('- patient_list = ' + s['fname_list'])
    logger.info('- fold = {}'.format(fold))
    logger.info('- patient range: [{} - {}]'.format(M, N))

    config['M'] = M
    metrics, metrics_str = test(test_loader, net, checkpoints_list, s, config, logger)  # actual testing

    logger.info('---------- SUMMARY ----------')
    header_str="Case_ID,WT_DSC,TC_DSC,ET_DSC"
    print(header_str)
    key_words = ['WT', 'TC', 'ET']

    if s['labeled']:
        print(metrics_str)  # content for outcsv_file

        if metrics.ndim == 1: metrics = np.expand_dims(metrics, 1)

        for i, key_word in enumerate(key_words):
            logger.info(
                'DICE {} : min = {:5.4f}, max = {:5.4f}, median = {:5.4f}, mean = {:5.4f}, std = {:5.4f}'.format(
                    key_word, np.nanmin(metrics[:, i]), np.nanmax(metrics[:, i]), np.nanmedian(metrics[:, i]),
                    np.nanmean(metrics[:, i]), np.nanstd(metrics[:, i])))

        f = open(outcsv_file, 'w+')
        f.write(header_str + '\n')
        f.write(metrics_str)
        f.close()

        logger.info('individual metrics are saved to ' + outcsv_file)

    elpased_time = time.time() - total_time_start
    logger.info('elapsed total time for predict {} cases took {:3.2f} s'.format(n_test, elpased_time))

if __name__ == '__main__':
    main()

