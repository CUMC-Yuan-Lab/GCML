import argparse, logging
import os, time
import numpy as np
import SimpleITK as sitk

from importlib import import_module
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from data.data import DP3DImageDataset

from utils import setgpu, count_parameters, set_logger, overlap_similarities

# only the best_val_loss model of each site is evaluated (the only checkpoint saved by gcml_brats_train.py)
MODEL = 'best_val_loss'

def get_net_and_checkpoints(c, model_loc):
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

    check_point = torch.load(model_loc)
    checkpoints_list = [check_point['state_dict']]
    print('loaded single {} model from {}'.format(model_name, model_loc))
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
        if c['save_mask']:
            mask_img = sitk.GetImageFromArray(relabled_mask)
            mask_img.CopyInformation(ref_img)

            mask_img_file = os.path.join(c['save_dir'], pt + '.nii.gz')  # used to be '_pred.nii.gz' in cross-validation
            sitk.WriteImage(mask_img, mask_img_file)
            print(mask_img_file + ' is saved')

        # save output files
        OARs = ['WT', 'TC', 'ET']
        if c['save_output']:
            i = 2
            output_img = sitk.GetImageFromArray(full_outputs[i])
            sitk.Cast(output_img, sitk.sitkFloat32)
            output_img.CopyInformation(ref_img)
            output_img_file = os.path.join(c['save_dir'], pt + '-' + OARs[i] + '.nii.gz')
            sitk.WriteImage(output_img, output_img_file)
            print(i + 1, output_img_file + ' is saved')

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
    parser.add_argument('--config', '-c', default='config_v0', type=str,
                        help='configuration file for model training/testing')
    parser.add_argument('M', type=int,
                        help='the starting case number (inclusive)')
    parser.add_argument('N', type=int,
                        help='the ending case number (inclusive)')
    parser.add_argument('--batch-size', '-b', default=1, type=int, metavar='N',
                        help='mini-batch size (default: 1)')
    parser.add_argument('-f', '--fold', default=0, type=int, metavar='F',
                        help='the fold used in training (default: 0)')
    parser.add_argument('-t', '--threshold', default=0.5, type=float,
                        help='threshold to the cnn output')
    parser.add_argument('--output_dir', type=str, metavar='OD', default='/mnt/data/brats2021/outputs',
                        help='the --output_dir used in gcml_brats_train.py: models are read from '
                             '<output_dir>/fl-params/<config>/, predictions are saved in <output_dir>/predictions/ '
                             'and logs in <output_dir>/logs/ (default: /mnt/data/brats2021/outputs)')
    parser.add_argument('--gpu', default='all', type=str, metavar='N',
                        help='the GPUs used (default: all)')
    parser.add_argument('--save_dir', type=str, metavar='SD',
                        help='directory to save prediction')
    parser.add_argument('-j', '--workers', default=8, type=int,
                        help='number of data loading workers (default: 8')
    parser.add_argument('--n_test', default=1, type=int,
                        help='number of VOI load for test')
    parser.add_argument('-s', '--save-output', action='store_true',
                        help='save the output of network before converting it to binary mask')
    parser.add_argument('--save_mask', action='store_true',
                        help='also save the predicted segmentation mask of every case (skipped by default, Dice '
                             'scores are always computed)')
    parser.add_argument('-l', '--log', action='store_true',
                        help='log: display and save debug info')
    parser.add_argument('--setting', default='SETTINGS_te', type=str,
                        help='the file specifying where data are read (default: SETTINGS_te), placed in settings/.')
    parser.add_argument('--reset', '-r', action='store_true',
                        help='clear saved result list and log file')
    parser.add_argument('--node',
                        help='the site whose best_val_loss model is used for prediction, e.g. 4, 5, 6, ...')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    total_time_start = time.time()

    print('loading config file: ' + args.config)
    tag = args.config[8:]
    config_training = import_module('configs.' + args.config)
    config = config_training.config
    config['tag'] = tag
    config['threshold'] = args.threshold
    config['save_output'] = args.save_output
    config['save_mask'] = args.save_mask
    config['args'] = args

    # logs are saved in <output_dir>/logs/
    output_dir = os.path.abspath(args.output_dir)
    hostpath = os.path.join(output_dir, 'logs') + '/'

    if not os.path.exists(hostpath):
        os.makedirs(hostpath)

    s = import_module('settings.' + args.setting)
    s = s.setting
    # model and output locations follow --output_dir instead of the settings file
    s['param_dir'] = os.path.join(output_dir, 'fl-params') + '/'
    s['output_dir'] = output_dir + '/'

    M = args.M
    N = args.N
    fold = args.fold
    node = args.node
    if node is None:
        raise ValueError('please specify --node')
    if node not in config['nodes']:
        raise ValueError('--node must be one of the sites {}, as only the best_val_loss model of each site is '
                         'saved'.format(config['nodes']))

    model_dir = os.path.join(s['param_dir'], args.config)

    # the only checkpoints written by gcml_brats_train.py: <node>_best_val_loss_f<fold>.ckpt
    model_fname = node + '_' + MODEL + '_f{}.ckpt'.format(fold)
    fcn_file = os.path.join(model_dir, model_fname)
    if not os.path.isfile(fcn_file):
        raise FileNotFoundError('model not found: ' + fcn_file)
    net, checkpoints_list = get_net_and_checkpoints(config, fcn_file)

    config['save_dir'] = args.save_dir
    if args.save_dir is None:
        config['save_dir'] = os.path.join(output_dir, 'predictions', args.config,
                                          node + '_' + MODEL + '_f{}'.format(fold))

    print('prediction results will be saved in :' + config['save_dir'])

    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])
    te_transform = None
    test_dataset = DP3DImageDataset(config, s, M, N, transform=te_transform)
    n_test = test_dataset.len()
    print('n_test = {}'.format(n_test))

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.workers,
                             pin_memory=True)

    outcsv_file = os.path.join(config['save_dir'],
                               'dice_tag{}_t{}_f{}.csv'.format(config['tag'], config['threshold'], fold))
    log_file = hostpath + config['prefix'] + "_te_v{}_f{}.txt".format(config['tag'], fold)
    if args.reset:
        if os.path.isfile(outcsv_file): os.remove(outcsv_file)
        # if os.path.isfile(log_file): os.remove(log_file)

    # add logger for cv only
    if 'prob_output' not in config.keys():
        config['prob_output'] = False

    if 'tta' not in config.keys():
        config['tta'] = False
    print('tta = {}'.format(config['tta']))

    logger = logging.getLogger(config['prefix'])

    logger = set_logger(logger, log_file, log=args.log)
    logger.info('*** start seg3d_predictV2 ({}: {}) ***'.format(args.setting, args.config))
    logger.info('- config = {}'.format(args.config))
    logger.info('- model = {}'.format(MODEL))
    logger.info('- threshold = {}'.format(args.threshold))
    logger.info('- patient_list = ' + s['fname_list'])
    logger.info('- node = {}'.format(node))
    logger.info('- model file = {}'.format(fcn_file))
    logger.info('- fold = {}'.format(fold))
    logger.info('- patient range: [{} - {}]'.format(M, N))

    config['M'] = M
    metrics, metrics_str = test(test_loader, net, checkpoints_list, s, config, logger)  # actual testing

    logger.info('---------- SUMMARY ----------')
    key_words = ['WT', 'TC', 'ET']

    if s['labeled']:
        print(metrics_str)  # content for outcsv_file [('WT', 1), ('TC', 2), ('ET', 3)]

        if metrics.ndim == 1: metrics = np.expand_dims(metrics, 1)

        for i, key_word in enumerate(key_words):
            logger.info(
                'DICE {} : min = {:5.4f}, max = {:5.4f}, median = {:5.4f}, mean = {:5.4f}, std = {:5.4f}'.format(
                    key_word, np.nanmin(metrics[:, i]), np.nanmax(metrics[:, i]), np.nanmedian(metrics[:, i]),
                    np.nanmean(metrics[:, i]), np.nanstd(metrics[:, i])))

        f = open(outcsv_file, 'w+')
        f.write(metrics_str)
        f.close()

        logger.info('individual dice coefficents are saved to ' + outcsv_file)

    elpased_time = time.time() - total_time_start
    logger.info('elapsed total time for predict {} cases took {:3.2f} s'.format(n_test, elpased_time))

if __name__ == '__main__':
    main()

