import argparse, logging
import os, sys, socket, time
import numpy as np
import SimpleITK as sitk

from importlib import import_module
import torch
import torch.nn.functional as F

from torch.nn import DataParallel
from torch.utils.data import DataLoader

from utils import setgpu, count_parameters, set_logger, sliding_window_inference_pred

from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize

from batchgenerators.utilities.file_and_folder_operations import load_pickle

from torch.utils.data import Dataset
import pandas as pd

from functools import partial
from SimpleITK import GetArrayViewFromImage as ArrayView

def refine_oar_mask(mask, ncls):
    output = np.zeros(sitk.GetArrayFromImage(mask).shape, dtype=np.uint8)
    for label in range(1,ncls):
        cc = sitk.ConnectedComponent(mask == label)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.Execute(cc, mask)
        l_sizes = np.asarray([stats.GetNumberOfPixels(l) for l in stats.GetLabels() if l != 0])
        max_l = np.argmax(l_sizes) + 1
        mask_l = sitk.ConnectedComponent(cc == max_l)

        output[np.nonzero(sitk.GetArrayFromImage(mask_l))] = label

    post_mask = sitk.GetImageFromArray(output)
    post_mask.CopyInformation(mask)
    post_mask = sitk.Cast(post_mask, sitk.sitkUInt8)
    return post_mask

def similarity_measures(pred, reference,label=1):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    pred = pred == label # sitk.ConnectedComponent(pred == label)
    reference = reference == label

    overlap_measures_filter.Execute(reference, pred)
    similarites = {}
    similarites['dice'] = overlap_measures_filter.GetDiceCoefficient()
    similarites['jaccard'] = overlap_measures_filter.GetJaccardCoefficient()
    similarites['voe'] = 1.0 - similarites['jaccard']
    similarites['rvd'] = 1.0 - overlap_measures_filter.GetVolumeSimilarity()

    distance_map = partial(sitk.SignedMaurerDistanceMap, squaredDistance=False, useImageSpacing=True)
    gold_surface = sitk.LabelContour(reference == label, False)
    prediction_surface = sitk.LabelContour(pred == label, False)

    prediction_distance_map = sitk.Abs(distance_map(prediction_surface))
    gold_distance_map = sitk.Abs(distance_map(gold_surface))

    gold_to_prediction = ArrayView(prediction_distance_map)[ArrayView(gold_surface) == 1]
    prediction_to_gold = ArrayView(gold_distance_map)[ArrayView(prediction_surface) == 1]

    similarites['hd95'] = (np.percentile(prediction_to_gold, 95) + np.percentile(gold_to_prediction, 95)) / 2.0
    similarites['assd'] = np.mean(list(prediction_to_gold) + list(gold_to_prediction))

    return similarites

def resampling(img, new_spacing, interp=sitk.sitkLinear):

    orig_sp = img.GetSpacing()
    orig_sz = img.GetSize()
    origin = img.GetOrigin()
    direct = img.GetDirection()
    px_type = img.GetPixelIDValue()

    print('orig_sp', orig_sp)
    print('new_sp', new_spacing)

    new_sz = [int(round(z1*p1/p2)) for z1,p1,p2 in zip(orig_sz, orig_sp, new_spacing)]
    resample_filter = sitk.ResampleImageFilter()

    resample_filter.SetOutputOrigin(direct)
    resample_filter.SetOutputOrigin(origin)
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetDefaultPixelValue(0)
    resample_filter.SetOutputPixelType(px_type)
    resample_filter.SetSize(new_sz)
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetInterpolator(interp)

    new_img = resample_filter.Execute(img)

    return new_img

def print_image_info(img, img_name):
    print(img_name)
    print('  size:      {}'.format(img.GetSize()))
    print('  origin:    {}'.format(img.GetOrigin()))
    print('  spacing:   {}'.format(img.GetSpacing()))
    print('  direction: {}'.format(img.GetDirection()))

def pt_pre_process3d(pt, s, c, model='train', save_preprocessed=False, take_pre_processed=False):
    start_time = time.time()
    print('*** pt_pre_process3d for patient {} ***'.format(pt))
    # resampled by the given spacing
    new_sp = c['imgsp']

    if take_pre_processed:
        print('take_pre_processed')
        # processed_dir = s['data_dir'] + 'v{}_{}x{}x{}/'.format(c['prep_ver'], new_sp[0], new_sp[1],
        #                                                        new_sp[2]) + 'imagesTr/'
        processed_dir = s['data_dir'] + 'imagesTr/'
        npy_pt = pt.split(sep='.')[0] + '.npy'
        full_npy_file = os.path.join(processed_dir, npy_pt)
        if os.path.isfile(full_npy_file):
            rsp_img_data = np.load(os.path.join(processed_dir, npy_pt))
            return rsp_img_data.squeeze(axis=0)
        assert(0)

    img_dir = s['orig_data_dir'] + 'imagesTr/'
    img = sitk.ReadImage(os.path.join(img_dir, pt))
    orig_direct = img.GetDirection()

    img_sp = np.round(img.GetSpacing(), decimals=4)
    # img_sz = img.GetSize()
    print_image_info(img, 'orig_image')
    img_data = sitk.GetArrayFromImage(img)
    print('img_data: ', img_data.min(), img_data.max())

    rsp_img = resampling(img, new_sp, interp=sitk.sitkBSpline) # interp=sitk.sitkLinear)
    rsp_img.SetDirection(orig_direct)

    print_image_info(rsp_img, 'resampled image')
    rsp_img_data = sitk.GetArrayFromImage(rsp_img).astype(np.float32)
    print('rsp_img_data: ', rsp_img_data.min(), rsp_img_data.max())
    # assert(0)
    # npy_pt = pt.split(sep='.')[0] + '.npy'
    if save_preprocessed:
        # processed_dir = s['data_dir'] + 'v{}_{}x{}x{}/'.format(c['prep_ver'], new_sp[0], new_sp[1], new_sp[2]) + 'imagesTr/'
        processed_dir = s['data_dir'] + 'imagesTr/'
        if not os.path.exists(processed_dir): os.makedirs(processed_dir)
        save_nrrd(rsp_img, os.path.join(processed_dir, pt), verbose=True)
        # np.save(os.path.join(processed_dir, npy_pt), np.expand_dims(rsp_img_data, axis=0))
    # assert(0)
    if model == 'train':
        label_dir = s['orig_data_dir'] + 'labelsTr/'
        labels = sitk.ReadImage(os.path.join(label_dir, pt))

        rsp_lbl = resampling(labels, new_sp, interp=sitk.sitkNearestNeighbor)
        rsp_lbl.SetDirection(orig_direct)
        print_image_info(rsp_lbl, 'resampled labels')
        # rsp_data = sitk.GetArrayFromImage(rsp_lbl)
        # print('rsp data: max = {}'.format(rsp_data.max()))
        print('   before resampling, labels = {}'.format(np.unique(sitk.GetArrayFromImage(labels))))
        print('   after resampling,  labels = {}'.format(np.unique(sitk.GetArrayFromImage(rsp_lbl))))
        if save_preprocessed:
            # processed_dir = s['data_dir'] + 'v{}_{}x{}x{}/'.format(c['prep_ver'], new_sp[0], new_sp[1], new_sp[2]) + 'labelsTr/'
            processed_dir = s['data_dir'] + 'labelsTr/'
            if not os.path.exists(processed_dir): os.makedirs(processed_dir)
            save_nrrd(rsp_lbl, os.path.join(processed_dir, pt), verbose=True)

        # assert(0)
        return
    else:
        return rsp_img_data

def save_nrrd(img, img_name, verbose=False):
    '''
    save compressed nrrd file
    :param img:
    :param img_name:
    :return:
    '''
    imgwriter = sitk.ImageFileWriter()
    imgwriter.SetFileName(img_name)
    imgwriter.SetUseCompression(True)
    imgwriter.Execute(img)
    if verbose:
        print('---')
        print(img_name + ' is saved.')
        print_image_info(img, 'info:')
        print('---')

class DP3DImageDataset(Dataset):
    def __init__(self, c, s, M, N, transform=None):

        super(DP3DImageDataset, self).__init__()

        self.c = c
        self.s = s
        # full_list = os.path.join(self.s['data_root'], self.s['fname_list'])
        full_list = self.s['fname_list']
        skip_rows = M - 1
        self.n_rows = N - M + 1
        df = pd.DataFrame(pd.read_csv(full_list, header=None, skiprows=skip_rows, nrows=self.n_rows))
        self.ptnames = df.values
        self.transform = transform

    def __getitem__(self, item):

        # pt = self.ptnames[item][0]
        pt = self.ptnames[item][0]
        print('pt = {}'.format(pt))

        #load image
        image = pt_pre_process3d(pt, s=self.s, c=self.c, model='test', save_preprocessed=False,
                                     take_pre_processed=self.s['labeled'])
        print('raw input_data', image.shape, image.min(), image.max())

        if self.c['one_hot_input']:
            ref_mask = image[1:].astype(np.int)
            n_refcls = len(self.c['img_ch'])
            ref_mask_one_hot = np.eye(n_refcls)[ref_mask.squeeze(axis=0)]
            ref_mask_one_hot = np.rollaxis(ref_mask_one_hot, -1, 0)
            assert (np.all(np.abs(np.argmax(ref_mask_one_hot, axis=0) - ref_mask) == 0))
            # print(ref_mask_one_hot.shape, image[0:1].shape)
            # assert(0)
            image = np.vstack((image[0:1], ref_mask_one_hot))

        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        image[0] = np.clip(image[0], a_min=self.c['lower_bound'], a_max=self.c['upper_bound'])
        print('after clip', image[0].min(), image[0].max())

        if self.transform is not None:
            image[0] = self.transform(image[0])

        if self.c['z_transform'] is not None:
            image[0] = (image[0] - self.c['z_transform'][0]) / self.c['z_transform'][1]
        else:
            image[0] = image[0] / self.c['upper_bound']

        voi = np.asarray([0, 0, 0] + list(image.shape[1:]))
        image = image.astype(np.float32)

        # assert(0)
        return (image, voi, pt)

    def __len__(self):
        return self.n_rows

    def len(self):
        return self.n_rows


def get_net_and_checkpoints(c, model_loc, ncv=1, mode='best_val_loss'):
    model_name = c['model']
    # in_channels = len(c['img_ch'])
    in_channels = len(c['img_ch']) + np.asarray(c['one_hot_input'])
    # if c['hq']:
    #     in_channels += 1

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
    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu

    if 'orl' not in c.keys():
        c['orl'] = ('orl' in c['loss'])

    # n_classes = 1 if len(c['OARs']) == 1 else len(c['OARs']) + 1
    n_classes = len(c['OARs']) + 1

    if c['orl'] and n_classes > 1:
        out_channels = n_classes - 1
    else:
        out_channels = n_classes

    if 'dp' not in c.keys():
        c['dp'] = False
    print('out_channels = {}'.format(out_channels))
    net = model.Net(in_channels=in_channels, nclasses=out_channels, nf=nf, relu=nonlin, up_mode=up_mode, dropout_p=dropout_p,
                        group_norm=gn, depth=depth, padding=True, deep_supervision=c['dp'])

    n_params = count_parameters(net)
    print(' - param_count = {0:,}'.format(n_params))
    # net = torch.compile(net)
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
        #print('loaded {} {} models in ensemble'.format(len(model_files), model_name)) # python 2
        print('loaded {} {} models in ensemble'.format(len(checkpoints_list), model_name)) #python 3
    return net, checkpoints_list

def roi_by_labels(mask_data, labels, min_sz, margin=0):
    assert(np.ndim(mask_data) == 3)
    if not isinstance(labels, list): labels = list(labels)
    label_mask = np.zeros_like(mask_data)
    for l in labels:
        label_mask[np.where(mask_data == l)] = 1
    idx = np.where(label_mask == 1)
    if len(idx[0]) > 0:
        starts =[]
        sz = []
        for i in range(3):
            print('dim[{}]: min = {}, max = {}'.format(i, idx[i].min(), idx[i].max()))
            idx_min = max(0, idx[i].min() - margin)
            idx_max = min(idx[i].max() + margin, mask_data.shape[i]-1)
            idx_sz = idx_max - idx_min + 1
            if idx_sz < min_sz[i]:
                idx_min = max(0, idx_min - int((min_sz[i] - idx_sz)/2))
                if idx_min + min_sz[i] > mask_data.shape[i]:
                    idx_min = mask_data.shape[i] - min_sz[i]
                idx_sz = min_sz[i]
            starts.append(int(idx_min))
            sz.append(int(idx_sz))
        voi = tuple(starts + sz)
        return voi
    else:
        return None

def test(data_loader, net, ckpt_list, s, c, logger):
    t0 = time.time()
    net.eval()
    metrics = []
    #metrics_str = 'pt,dice\n'
    metrics_str = ''

    M = c['M']
    N = c['N']
    fold = c['fold']
    outcsv_file = os.path.join(c['save_dir'], 'dice_tag{}_f{}.csv'.format(c['tag'], fold))

    # cx = c['crop_szx']
    # cy = c['crop_szy']
    cz = np.asarray([c['crop_szz'], c['crop_szy'], c['crop_szx']])
    nch = len(c['img_ch'])
    ncls= len(c['OARs'])+1

    if c['orl']:
        nonlin = lambda x: F.sigmoid(x)
        ncls -= 1
    else:
        nonlin = lambda x: F.softmax(x, dim=1)

    importance_map = 1.0

    for i, (data, voi, pt) in enumerate(data_loader):
        # print('data.type = {}, size = {}'.format(type(data), data.size()))
        data = torch.squeeze(data, dim=0)
        voi = torch.squeeze(voi)
        pt = pt[0]
        print('data.size = {}'.format(data.size()))
        iz = data.size()[1:]
        print('voi = {}'.format(voi))
        outstr_i = 'pt{:04d},{}'.format(M+i, pt)
        full_outputs = np.zeros((ncls, iz[0], iz[1], iz[2]), dtype=np.float32)

        patch_outputs = sliding_window_inference_pred(data, voi, net, patch_size=cz, ncls=ncls, step_ratio=0.5,
                                                      importance_map=importance_map, nonlin=nonlin, ckpt_list=ckpt_list,
                                                      amp=c['amp'], tta=c['tta'], verbose=False)

        print('patch_outputs.shape: ', patch_outputs.shape)
        full_outputs[:,voi[0]:voi[0]+voi[3], voi[1]:voi[1]+voi[4], voi[2]:voi[2]+voi[5]] = patch_outputs

        orig_img = sitk.ReadImage(os.path.join(s['orig_data_dir'] + '/imagesTr', pt))
        raw_size = sitk.GetArrayFromImage(orig_img).shape # .astype(np.float32)

        # pkl_dir = s['data_dir'] + 'v{}_{}x{}x{}'.format(c['prep_ver'], c['imgsp'][0], c['imgsp'][1], c['imgsp'][2]) + '/imagesTr/'
        pkl_dir = s['data_dir'] + '/imagesTr/'
        pkl_props = load_pickle(os.path.join(pkl_dir, pt.split('.')[0] + '.pkl'))
        bbox = pkl_props['bbox_used_for_cropping']
        bbox_size = [bbox[0][1]-bbox[0][0], bbox[1][1]-bbox[1][0], bbox[2][1]-bbox[2][0]]

        curr_size = full_outputs.shape[1:] #.astype(np.float32)
        # zoom_ratio = np.asarray([orig_size[d]/curr_size[d] for d in range(3)])
        n_output = full_outputs.shape[0]
        full_outputs2 = np.zeros([n_output] + list(bbox_size), dtype=np.float32)
        separate_z = False

        if separate_z:
            for ch in range(n_output):
                print('ch = ', ch)
                data_c = []
                for z in range(curr_size[0]): # z_direction
                    data_c.append(resize(full_outputs[ch, z], bbox_size[1:], order=1, mode='edge', anti_aliasing=False))
                data_c = np.stack(data_c, axis=0)
                full_outputs2[ch] = resize(data_c, bbox_size, order=0,  mode='edge', anti_aliasing=False)
        else:
            for ch in range(n_output):
                print('ch = ', ch)
                full_outputs2[ch] = resize(full_outputs[ch], bbox_size, order=1, mode='edge', anti_aliasing=False)

        full_outputs = full_outputs2

        if c['orl']:
            temp = np.zeros_like(full_outputs).astype(np.uint8)
            temp[np.where(full_outputs > 0.5)] = 1
            full_outputs = temp.squeeze()
        else:
            full_outputs = np.argmax(full_outputs, axis=0).astype(np.uint8)

        # put full_outputs back to bbox

        full_outputs3 = np.zeros(raw_size, dtype=np.uint8)
        full_outputs3[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = full_outputs
        full_outputs = full_outputs3

        pred_mask = sitk.GetImageFromArray(full_outputs)
        pred_mask.CopyInformation(orig_img)
        pred_mask = refine_oar_mask(pred_mask, ncls)

        pred_mask = sitk.Cast(pred_mask, sitk.sitkUInt8)
        save_nrrd(pred_mask, os.path.join(c['save_dir'], pt))
        print('postprocessed mask prediction for {} is saved.'.format(pt))
        # assert(0)

        if s['labeled']:
            overlaps = []
            pt=pt.split('.')[0] # remove file extension .nii.gz
            metrics_str += pt
            label_file = s['orig_data_dir'] + 'labelsTr/' + pt  # raw label file is loaded
            OAR_label = sitk.ReadImage(label_file)

            OAR_label = sitk.Cast(OAR_label, sitk.sitkUInt8)
            pred_mask_data = sitk.GetArrayFromImage(pred_mask)
            oar_label_data = sitk.GetArrayFromImage(OAR_label)
            for OAR, label in sorted(c['OARs'].items(), key=lambda x: x[1]):

                #compute similarity inculding dice
                scores = similarity_measures(pred_mask, OAR_label, label)
                dice = scores['dice']

                if dice == np.inf:
                    pred_size = len(np.where(pred_mask_data==label)[0])
                    gt_size = len(np.where(oar_label_data == label)[0])
                    print('pred_dize = {}, gt_size = {}'.format(pred_size, gt_size))
                    if gt_size == 0 and pred_size == 0:
                        dice = 1.0
                    else:
                        dice = 0.0
                    # dice = np.nan
                print('    OAR = {}, label = {}, DICE = {:.4f},'.format(OAR, label, dice))
                overlaps.append(dice)
                metrics_str += ',{:5.4f}'.format(dice)
                outstr_i += ',{:5.4f}'.format(dice)

                # extract HD and ASSD
                hd = scores['hd95']
                overlaps.append(hd)
                metrics_str += ',{:5.4f}'.format(hd)
                outstr_i += ',{:5.4f}'.format(hd)

                assd = scores['assd']
                overlaps.append(assd)
                metrics_str += ',{:5.4f}'.format(assd)
                outstr_i += ',{:5.4f}'.format(assd)

            metrics.append(overlaps)
            metrics_str += '\n'
            outstr_i += '\n'
            print(outstr_i)
            # assert(0)
            f = open(outcsv_file, 'a+')
            f.write(outstr_i)
            f.close()

    metrics = np.asarray(metrics)
    elpased_time = time.time() - t0
    print('predict {} cases took {:3.2f} s'.format(i+1, elpased_time))
    # print metrics
    return metrics, metrics_str

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch 2D Deep Planning: Dose prediction')
    parser.add_argument('-f', '--fold', type=int, metavar='F',
                    help='the models from cross-validation training')
    parser.add_argument('--gpu', default='all', type=str, metavar='N',
                        help='the GPUs used (default: all)')
    parser.add_argument('-m', '--model', default='best_val_loss', choices=['best_val_loss', 'best_val_loss_ma', 'history'],
                        help='the model used for prediction,, chosen from [best_val_loss, best_val_loss_ma, history]')
    total_time_start = time.time()
    args = parser.parse_args()

    config_str = "config_vpanseg_demo"
    tag = config_str[8:]
    config_training = import_module('configs.' + config_str)
    config = config_training.config
    config['tag'] = tag
    config['threshold'] = 0.5

    s = import_module('settings.SETTINGS_te')
    s = s.setting

    M = 1
    N = 76
    fold = args.fold

    model_dir = os.path.join(s['param_dir'], config_str)
    config['one_hot_input'] = (config['prep_ver'] == 2 and len(config['img_ch']) > 2)

    if fold is None:
        net, checkpoints_list = get_net_and_checkpoints(config, model_dir, ncv=config['NCV'], mode=args.model)
    else:
        model_fname = args.model + '_f{}'.format(fold)
        model_fname += '.ckpt'
        fcn_file = os.path.join(model_dir, model_fname)
        net, checkpoints_list = get_net_and_checkpoints(config, fcn_file, ncv=1)


    config['save_dir'] = config_str + '/' + args.model
    if args.fold is not None:
        config['save_dir'] = config['save_dir'] + '-f{}'.format(fold)

    config['save_dir'] = os.path.join(s['output_dir'], config['save_dir'])

    print('predicted mask will be saved in :' + config['save_dir'])
    # assert(0)

    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])
    te_transform = None

    test_dataset = DP3DImageDataset(config, s, M, N, transform=te_transform)

    n_test = test_dataset.len()

    test_loader = DataLoader(test_dataset,
                              batch_size = 1,
                              shuffle = False,
                              num_workers = 8,
                              pin_memory=True)


    outcsv_file = os.path.join(config['save_dir'], 'dice_tag{}_f{}.csv'.format(config['tag'], fold))
    log_file = config['save_dir'] + config['prefix'] + "_te_v{}_f{}.txt".format(config['tag'], fold)

    if os.path.isfile(outcsv_file): os.remove(outcsv_file)

    # add logger for cv only
    if 'prob_output' not in config.keys():
        config['prob_output'] = False

    if 'tta' not in config.keys():
        config['tta'] = False
    if 'z_transform' not in config.keys():
        config['z_transform'] = None
    if 'copy_label' not in config.keys():
        config['copy_label'] = True
    if 'use_gaussian' not in config.keys():
        config['use_gaussian'] = False
    if 'amp' not in config.keys():
        config['amp'] = False

    # determine if it's pelvic region only
    if len(config['OARs']) == 2 and 'bladder' in config['OARs'] and 'prostate' in config['OARs']:
        config['pelvic_only'] = True
    else:
        config['pelvic_only'] = False

    logger = logging.getLogger(config['prefix'])

    logger = set_logger(logger, log_file, log=False)


    config['M'] = M
    config['N'] = N
    config['fold'] = fold

    metrics, metrics_str = test(test_loader, net, checkpoints_list, s, config, logger)

    logger.info('---------- SUMMARY ----------')
    logger.info('- patient_list = ' + s['fname_list'])
    logger.info('- fold = {}'.format(fold))
    logger.info('- patient range: [{} - {}]'.format(M, N))

    if s['labeled']:
        header_str = "Case_ID,DSC,HD95,ASSD"
        print(header_str)
        print(metrics_str)
