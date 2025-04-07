import argparse, logging
import os, sys, time, bisect, shutil
import numpy as np

from tqdm import tqdm
from importlib import import_module

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

sys.path.append('../')

from utils import setgpu, count_parameters, set_logger, init_weights, print_stdout_to_file,pt_pre_process, img_augument3d, voi_histogram_equalization

import pandas as pd
from torch.utils.data import Dataset

class myH5PYDatasetSeg3D(Dataset):
    def __init__(self, s, c, phase='train', folds=None, n_classes=1):

        super(myH5PYDatasetSeg3D, self).__init__()
        assert (phase == 'train' or phase == 'test')

        self.c = c
        self.s = s
        full_list = self.s['fname_list']
        df = pd.DataFrame(pd.read_csv(full_list, header=None))
        ptnames = [df.values[i][0] for i in range(len(df.values))]
        npt = len(ptnames)
        n_cv = folds[0]
        fold_n = npt // n_cv

        f = folds[1]
        if f == 0:
            te_ptnames = ptnames[0:fold_n]
            tr_ptnames = ptnames[fold_n:]
        elif f > 0 and f < n_cv - 1:
            te_ptnames = ptnames[f*fold_n:(f+1)*fold_n]
            tr_ptnames = ptnames[0:f*fold_n] + ptnames[(f+1)*fold_n:]
        elif f == n_cv - 1:
            te_ptnames = ptnames[f*fold_n:]
            tr_ptnames = ptnames[0:f*fold_n]
        else:
            te_ptnames = ptnames
            tr_ptnames = ptnames

        crop = (c['img_szz'] - c['crop_szz'], c['img_szy'] - c['crop_szy'], c['img_szx'] - c['crop_szx'],
                c['crop_szz'], c['crop_szy'], c['crop_szx'])

        if phase == 'train':
            self.ptnames = tr_ptnames
            if c['augument_fcn'] is 'img_augument3d':
                self.transform = lambda x: img_augument3d(x, crop=crop, shift=c['shift'], rotate=c['rotate'], scale=c['scale'],
                                                  normalize_pctwise=c['pct_norm_tr'],
                                                  p_ud=c['p_ud'], p_lr=c['p_lr'], p_si=c['p_si'], p_rot90=c['p_rot90'],
                                                  contrast=c['contrast'],
                                                  gamma = c['gamma'],
                                                  istest=False, mrgn=c['margin'])
            else:
                raise ValueError(c['augument_fcn'] + ' is not implemented')
        else:
            self.ptnames = te_ptnames
            if c['augument_fcn'] is 'img_augument3d':
                self.transform = lambda x: img_augument3d(x, crop=crop, normalize_pctwise=c['pct_norm'], istest=True)
            else:
                raise ValueError(c['augument_fcn'] + ' is not implemented')

        self.n_classes = n_classes
        self.c = c
        self.weight_sample = c['weight_sample']

        self.n_data = len(self.ptnames)

        self.phase = phase


    def __getitem__(self, idx):
        pt = self.ptnames[idx]

        image, label = pt_pre_process(pt, s=self.s, c=self.c, model='train')
        image = np.squeeze(image)
        label = np.squeeze(label)

        # in order to use batchgenerator package, change image dim to [ch, szx,szy,szz], and label dim to [1, szx,szy,szz]

        image = np.rollaxis(image, 3, 0).astype(np.float32) # [c,z,y,x]

        image = np.swapaxes(image, 1, 3)  # [c,x,y,z]

        #add hq channel if c['hq'] == True, legacy setup, not use anymore YY 06/18/23
        if self.c['hq']:
            t1ce = image[1]
            br_mask = np.zeros_like(t1ce)
            br_mask[np.where(t1ce>0)] = 1
            hq_t1ce = np.expand_dims(voi_histogram_equalization(t1ce, br_mask, number_bins=256),axis=0).astype(np.float32)
            image = np.vstack((image, hq_t1ce))

        label = np.expand_dims(np.swapaxes(label, 0, 2), axis=0)

        label_max = label.max()
        if self.transform:
            sample = (image, label)
            image, label = self.transform(sample)
        label_crop_max = label.max()
        ncls = self.n_classes

        if self.weight_sample is not None:
            weight = np.asarray(self.weight_sample, dtype=np.float32)
        else:
            if ncls > 1:
                n_pixels = label.shape[1] * label.shape[2] * label.shape[3]
                weight = 1.0 * np.asarray([np.count_nonzero(label == i) for i in range(ncls)], dtype=np.float32)
                weight[np.nonzero(weight)] = n_pixels / weight[np.nonzero(weight)]
                weight = weight / weight.sum()

            else:
                weight = 1.0

        if self.c['orl']:
            label = np.squeeze(label)

            label_new = np.zeros((ncls-1, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
            for cls in range(1, ncls):
                for j in range(cls):
                    temp_j = np.zeros(label.shape)
                    temp_j[np.where(label==cls)] = 1
                    label_new[j] += temp_j

            temp = np.sum(label_new, axis=0)

            assert(np.sum(np.abs(label - temp)) == 0)
            label = label_new
            if self.c['weight_sample'] is not None:
                weight = np.asarray(self.c['weight_sample'], dtype=np.float32)
            else:
                weight = 1.0 * np.asarray([np.count_nonzero(label[i] == 1) for i in range(ncls-1)], dtype=np.float32)
                cls_sum = weight.sum()
                weight[np.nonzero(weight)] = cls_sum / weight[np.nonzero(weight)]
                # YY added on 6/19/20, if ET is missing, assign its weight as TC
                if label_max < 3:
                    weight[np.where(weight==0)] = weight[np.nonzero(weight)].max() * 2.0
                weight = weight / weight.sum()

        return (image, label, weight)

    def __len__(self):
        return self.n_data

    def len(self):
        return self.n_data

def get_loss(loss_name, dp = 0):
    '''
    return loss function
    :param loss_name:
    :param dp: if deep supervision is enabled
    :return:
    '''
    if loss_name in ['bja_orl', 'bja']:
        criterion = BinaryJaccardLoss()
    elif loss_name in ['bja_bfl_orl', 'bja_bfl']:
        criterion = BinaryJaccardAndFocalLoss()
    else:
        err_msg = loss_name + ' is not implemented yet!'
        raise ValueError(err_msg)
    if dp == 0:
        return criterion
    elif dp == 1:
        return DeepSupervision(criterion=criterion)
    elif dp == 2:
        return DeepSupervisionV2(criterion=criterion)
    else:
        ValueError('dp = {} is not implemented, currently support dp = [0,1,2]'.format(dp))

class DeepSupervision(nn.Module):
    def __init__(self, criterion):
        super(DeepSupervision, self).__init__()
        self.criterion = criterion
    def forward(self, input, target, weight=[0.5, 0.5], ss=10):
        assert(isinstance(input, list))
        n_inputs = len(input)
        loss0 = lossdp = 0
        #sum_i = 0
        if n_inputs > 1:
            for i in range(n_inputs):
                loss_i = self.criterion(input[i], target, weight, ss)
                if i == 0:
                    loss0 = loss_i[0]
                else:
                    lossdp += loss_i[0] # (n_inputs - i) * loss_i[0] # loss_i[0]

            lossdp = lossdp / (n_inputs - 1)
            # loss = 0.5 * loss0 + 0.5 * lossdp # used for brats 20
            loss = 0.4 * loss0 + 0.6 * lossdp # YY test on 02/09/21, used for brats 21
            return [loss, loss0.item(), lossdp.item()]
        else:
            loss = self.criterion(input[0], target, weight, ss)
            return loss

class DeepSupervisionV2(nn.Module):
    def __init__(self, criterion):
        super(DeepSupervisionV2, self).__init__()
        self.criterion = criterion
    def forward(self, input, target, weight=[0.5, 0.5], ss=10):
        assert(isinstance(input, list))
        n_inputs = len(input)
        loss0 = lossdp = 0

        sum_i = 0
        if n_inputs > 1:
            for i in range(n_inputs):
                loss_i = self.criterion(input[i], target, weight, ss)
                if i == 0:
                    loss0 = loss_i[0]
                else:
                    lossdp += (n_inputs - i) * loss_i[0]
                sum_i += (n_inputs - i)

            lossdp = lossdp / (sum_i - n_inputs)
            loss = (n_inputs * loss0 + (sum_i - n_inputs) * lossdp) / sum_i
            return [loss, loss0.item(), lossdp.item()]
        else:
            loss = self.criterion(input[0], target, weight, ss)
            return loss

class BinaryJaccardLoss(nn.Module):
    def __int__(self):
        super(BinaryJaccardLoss, self).__init__()
        # self.ss = ss

    def forward(self, input, target, weight = None, ss=10):
        '''
        with weight_i = 1/sum(c_i)
        :param input: (batch_size, 1, szy, szx)
        :param target: (batch_size, 1, szy, szx)
        :return:
        '''
        # print(input.shape, target.shape, type(input), type(target))
        assert(input.shape == target.shape)
        #print('ss = {}'.format(ss))
        #assert(0)
        #input = F.sigmoid(input)
        target = target.type(torch.cuda.FloatTensor)
        nb = target.size(0) # batch-size
        nc = target.size(1) # # of classes

        target = target.view(nb, nc, -1)
        input = input.view(nb, nc, -1)

        intersection = input * target
        # ss0 = 1.0
        #print('ss0 = {}'.format(ss0))
        #assert(0)
        jaccard = 1.0 - (intersection.sum(dim=2) + ss) / (input.sum(dim=2) + target.sum(dim=2) - intersection.sum(dim=2) + ss)
        # print('weight',weight)
        # print('jaccard',jaccard)
        # print weight
        # assert(0)

        if weight is not None:
            # print weight
            jaccard = jaccard * weight
            # print('w-jaccard',jaccard)
            loss = jaccard.sum(dim=-1).mean(dim=0)
        else:
            loss = jaccard.mean()
        # print(dice.shape)
        jaccard = jaccard.mean(dim=0)
        # print('f-jaccard',jaccard)

        # print(loss)
        # assert(0)
        # return [loss, jaccard[0].item(), jaccard[1].item(), jaccard[2].item()]
        return [loss] + [jaccard[i].item() for i in range(nc)]

class MyFocalLoss(nn.Module):
    def __int__(self, alpha=1, gamma=2):
        super(MyFocalLoss, self).__init__()
        #self.alpha = alpha
        #self.gamma = gamma

    def forward(self, input, target, weight=None):
        '''

        :param input: (batch_size, n_classes, szy, szx)
        :param target: (batch_size, szy, szx)
        :return:
        '''
        #print weight
        # print input.size(), target.size()
        #assert(0)
        # criterion = nn.BCEWithLogitsLoss()

        #logit = F.sigmoid(input)
        #logit = logit.clamp(1e-7,1-1e-7)

        bce_loss = F.binary_cross_entropy(input, target, reduce=False)
        pt = torch.exp(-1*bce_loss)
        # FL_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        FL_loss = ((1 - pt) ** 2) * bce_loss
        nb = input.size(0)
        nc = input.size(1)
        FL_loss = FL_loss.view(nb,nc,-1).mean(dim=2)
        # print('FL_loss', FL_loss)
        if weight is not None:
            # print(weight)
            FL_loss = FL_loss * weight
            FL_loss = FL_loss.sum(dim=-1).mean(dim=0)
        else:
            FL_loss = FL_loss.mean()
            # print('fl: no weight')
            #assert(0)
        # print('final FL_loss', FL_loss)
        # print(FL_loss.size())
        # assert(0)
        #loss = FL_loss.mean()

        return [FL_loss]

class BinaryJaccardAndFocalLoss(nn.Module):
    def __int__(self):
        super(BinaryJaccardAndFocalLoss, self).__init__()
        # self.ss = ss

    def forward(self, input, target, weight = None, ss=10):
        '''
        with weight_i = 1/sum(c_i)
        :param input: (batch_size, 1, szy, szx)
        :param target: (batch_size, 1, szy, szx)
        :return:
        '''
        # print(input.shape, target.shape, type(input), type(target))
        # one-hot
        criterion_bja = BinaryJaccardLoss()
        criterion_fl = MyFocalLoss()
        #bja = criterion_bja(F.sigmoid(input), target, weight, ss)
        #print('bja+bfl: weight = {}, ss = {}'.format(weight, ss))
        bja = criterion_bja(input, target, weight, ss)
        fl = criterion_fl(input, target, weight=None)
        loss = 0.5 * bja[0] + 0.5 * fl[0]
        #loss = 1.0 * bja[0] + 0.5 * fl[0]
        nc = target.size(1)
        return [loss, bja[0].item(), fl[0].item()] + [bja[i] for i in range(1, nc+1)]

def train(data_loader, net, loss, c, optimizer, lr, amp_grad_scaler=None):

    net.train()

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    metrics = []

    output_prob = []
    clss = []
    weights = []
    for (data, target, weight) in tqdm(data_loader):

        cls = np.max(target.numpy(), axis=(2,3,4))[:,-1]

        if c['weight_sample'] is None:
            weights.append(weight.numpy())

        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        weight = weight.cuda(non_blocking=True)
        target = target.type(torch.cuda.FloatTensor)
        weight = weight.type(torch.cuda.FloatTensor)

        # automatic mixed precision
        optimizer.zero_grad()
        if c['amp']:
            with autocast():
                output = net(data)
                if (c['prob_output'] is False) and ('logits' not in c['loss']):
                    assert(isinstance(output, list))
                    for i in range(len(output)):
                        output[i] = F.sigmoid(output[i]).clamp(1e-7, 1-1e-7)

                if isinstance(output, list) and c['dp'] is False:
                    loss_output = loss(output[0], target, weight)
                else:
                    loss_output = loss(output, target, weight)

            amp_grad_scaler.scale(loss_output[0]).backward()
            if c['grad_norm_clip'] is not None :
                amp_grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), c['grad_norm_clip'])
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
        else:
            output = net(data)
            if (c['prob_output'] is False) and ('logits' not in c['loss']):
                assert(isinstance(output, list))
                for i in range(len(output)):
                    output[i] = F.sigmoid(output[i]).clamp(1e-7, 1-1e-7)

            if isinstance(output, list) and c['dp'] is False:
                loss_output = loss(output[0], target, weight)
            else:
                loss_output = loss(output, target, weight)

            loss_output[0].backward()
            if c['grad_norm_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), c['grad_norm_clip'])

            optimizer.step()

        loss_output[0] = loss_output[0].item()

        metrics.append(loss_output)

        if 'logits' in c['loss']:
            output[0] = F.sigmoid(output[0])
        et_prob = np.max(output[0].data.cpu().numpy(), axis=(2, 3, 4))[:,-1]

        if isinstance(cls, np.ndarray):
            for i in range(len(cls)):
                output_prob.append(et_prob[i])
                clss.append(cls[i])
        else:
            output_prob.append(et_prob)
            clss.append(cls)

    metrics = np.asarray(metrics, np.float32)

    train_loss = np.mean(metrics, axis=0)

    # Below is specifically designed for BraTS challenge.
    # As some cases don't have enhanced tumor (ET), DICE value will be zero if the model incorrectly generate ET for
    # these cases, or vice versa. So the following code is used to monitor how many cases are correctly classified as
    # with or without ET

    clss = np.asarray(clss, np.uint8)
    nclss = len(clss)
    # print('clss', clss.shape, nclss)
    if nclss > 0:
        output_prob = np.asarray(output_prob, np.float32)
        output_cls = np.zeros_like(output_prob, dtype=np.uint8)
        output_cls[np.where(output_prob>=0.5)] = 1
        pos_idx = np.where(clss==1)
        neg_idx = np.where(clss==0)
        cls_results = ~np.logical_xor(output_cls, clss)
        correct_pos = np.sum(cls_results[pos_idx])
        n_pos = np.count_nonzero(clss)
        correct_neg = np.sum(cls_results[neg_idx])
        # correct_cls = np.sum(~np.logical_xor(output_cls, clss))
        prt_str = '[{}/{}] positive and [{}/{}] negative training cases are correctly classified'.format(correct_pos, n_pos,
                                                                                                     correct_neg, nclss-n_pos)
        if c['weight_sample'] is None:
            weights = np.asarray(weights, np.float32)
            prt_str += ', average weights = {}'.format(np.mean(weights, axis=0))

        # print(prt_str)

    if 'fs' in c['loss']:
        fs = metrics[:,-1]
        print('fs: min = {:.5f}, max = {:.5f}, median = {:.5f}, mean = {:.5f}'.format(fs.min(), fs.max(),
                                                                                      np.median(fs),fs.mean()))

    return train_loss, prt_str

def validate(data_loader, net, loss, c, amp_grad_scaler=None):

    net.eval()
    metrics = []

    output_prob = []
    clss = []
    with torch.no_grad():
        for i, (data, target, weight) in enumerate(data_loader):

            cls = np.max(target.numpy(), axis=(2, 3, 4))[:, -1]
            data = data.cuda()
            target = target.cuda()
            target = target.type(torch.cuda.FloatTensor)

            # automatic mixed precision
            if c['amp']:
                with autocast():
                    output = net(data)
                    if (c['prob_output'] is False) and ('logits' not in c['loss']):
                        assert (isinstance(output, list))
                        for i in range(len(output)):
                            output[i] = F.sigmoid(output[i]).clamp(1e-7, 1-1e-7)

                    if isinstance(output, list) and c['dp'] is False:
                        loss_output = loss(output[0], target, weight=None)
                    else:
                        loss_output = loss(output, target, weight=None)
            else:
                output = net(data)
                if (c['prob_output'] is False) and ('logits' not in c['loss']):
                    assert (isinstance(output, list))
                    for i in range(len(output)):
                        output[i] = F.sigmoid(output[i]).clamp(1e-7, 1-1e-7)


                if isinstance(output, list) and c['dp'] is False:
                    loss_output = loss(output[0], target, weight=None)
                else:
                    loss_output = loss(output, target, weight=None)

            loss_output[0] = loss_output[0].item()
            metrics.append(loss_output)
            if 'logits' in c['loss']:
                output[0] = F.sigmoid(output[0])
            et_prob = np.max(output[0].data.cpu().numpy(), axis=(2, 3, 4))[:,-1]
            if isinstance(cls, np.ndarray):
                for i in range(len(cls)):
                    output_prob.append(et_prob[i])
                    clss.append(cls[i])
            else:
                output_prob.append(et_prob)
                clss.append(cls)


    metrics = np.asarray(metrics, np.float32)
    valid_loss = np.mean(metrics, axis=0)

    # below is specifically designed for BraTS challenge
    clss = np.asarray(clss, np.uint8)
    nclss = len(clss)
    if nclss > 0:
        output_prob = np.asarray(output_prob, np.float32)
        output_cls = np.zeros_like(output_prob, dtype=np.uint8)
        output_cls[np.where(output_prob >= 0.5)] = 1
        pos_idx = np.where(clss == 1)
        neg_idx = np.where(clss == 0)
        cls_results = ~np.logical_xor(output_cls, clss)
        correct_pos = np.sum(cls_results[pos_idx])
        n_pos = np.count_nonzero(clss)
        correct_neg = np.sum(cls_results[neg_idx])
        # correct_cls = np.sum(~np.logical_xor(output_cls, clss))
        prt_str = '[{}/{}] positive and [{}/{}] negative validating cases are correctly classified'.format(correct_pos, n_pos,
                                                                                                     correct_neg,
                                                                                                     nclss - n_pos)

    return valid_loss, prt_str

def get_args():

    # global args
    parser = argparse.ArgumentParser(description='PyTorch 3D Deep Planning')
    parser.add_argument('-f', '--fold', type=int, metavar='F',
                        help='the fold for cross-validation when training models')
    parser.add_argument('--gpu', default='0', type=str, metavar='N',
                        help='the GPUs used (default: all)')
    parser.add_argument('--setting', default='SETTINGS_train_allsites', type=str, choices=['SETTINGS_train_allsites','SETTINGS_train_8sites'],
                        help='sites included in training (default: SETTINGS_train_allsites)')
    parser.add_argument('--config', '-c', default='config_v0', type=str,
                        help='configuration file for model training/testing, placed in configs/.')
    args = parser.parse_args()
    return args


def main():

    # get global args from command line input
    args = get_args()

    # log file will be saved in the folder named as the hostname
    hostpath = './log/'
    if not os.path.exists(hostpath):
        os.makedirs(hostpath)

    # load setting
    s = import_module('settings.' + args.setting)
    s = s.setting

    # load configuration

    tag = args.config[8:]
    config_training = import_module('configs.' + args.config)
    c = config_training.config
    c['tag'] = tag

    f = args.fold  # cross validation on fold f
    NCV = c['NCV']
    assert (f >= 0 and f <= NCV)

    # create log file
    logger = logging.getLogger(c['prefix'])
    log_file = hostpath + c['prefix'] + '_tr_v{}_f{}.txt'.format(c['tag'], f)

    logger = set_logger(logger, log_file, log=False)

    in_channels = len(c['img_ch'])

    # set up parameters for model
    model_name = c['model']
    dropout_p = c['dropout_p']
    gn = c['group_norm']
    up_mode = c['up_mode']
    depth = c['depth']
    nf = c['nf']
    n_classes = len(c['OARs']) + 1
    if 'orl' not in c.keys():
        c['orl'] = ('orl' in c['loss'])

    # backward compatible with early version of configuration files
    if 'nonlin' not in c.keys():
        c['nonlin'] = 'relu'
    nonlin = c['nonlin']
    if nonlin == 'relu':
        nonlin_w = 'relu'
    else:
        nonlin_w = 'leaky_relu'

    # if deep supervision will be used
    if 'dp' not in c.keys():
        c['dp'] = 0
    dp = int(c['dp'])

    # model initialization method
    if 'init' not in c.keys():
        c['init'] = 'xavier'
    init = c['init']

    # if moving average will be used when evaluating validation loss
    if 'val_ma_alpha' not in c.keys():
        c['val_ma_alpha'] = 0.0
    val_ma_alpha = c['val_ma_alpha']

    # lr_decay_strategy
    if 'lr_decay' not in c.keys():
        c['lr_decay'] = 'step'
    lr_decay = c['lr_decay']

    # automatic mixed precision (AMP)
    if 'amp' not in c.keys():
        c['amp'] = False
    if c['amp']:
        amp_grad_scaler = GradScaler()
    else:
        amp_grad_scaler = None

    out_channels = n_classes
    if c['orl']:
        out_channels -= 1
    if 'prob_output' not in c.keys():
        c['prob_output'] = False

    # create DL model and initialize weights
    model = import_module('models.' + model_name)
    net = model.Net(in_channels=in_channels, nclasses=out_channels, nf=nf, relu=nonlin, up_mode=up_mode, dropout_p=dropout_p,
                        depth=depth, padding=True, group_norm=c['group_norm'], deep_supervision=c['dp'])

    init_weights(net, init_type=init, nonlin=nonlin_w)

    # set the path to save model weights
    save_dir = os.path.join(s['param_dir'], args.config)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # the stdout on the terminal will be saved in a stdout file in the parameter folder
    stdout_file = os.path.join(save_dir, 'stdout_f{}.txt'.format(f))

    # set GPU and move network model and loss functions to GPU
    loss = get_loss(c['loss'], dp=dp)
    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    net = net.cuda()
    loss = loss.cuda()
    # val_loss = loss if 'val_loss' is not specified in configuration
    if 'val_loss' not in c.keys():
        c['val_loss'] = c['loss']
        val_loss = loss
    else:
        val_loss = get_loss(c['val_loss'], dp=dp)
        val_loss = val_loss.cuda()

    # vloss_str = c['val_loss']

    n_params = count_parameters(net)
    logger.info(' - param_count = {0:,}'.format(n_params))
    logger.info('trained models and stdout will be saved in :' + save_dir)

    # data parallel is used to distribute data to train a model on multiple GPUs (when batch_size > 1 and n_gpu > 1)
    net = DataParallel(net)

    lr_0 = 0.003
    # set optimization methods
    if c['optim'] is 'adam':
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr = lr_0,
            weight_decay = 0,
            amsgrad=True
        )

    optim_str = c['optim']

    num_epochs = 200
    start_epoch = 0
    best_loss_epoch = num_epochs
    best_loss_MA_epoch = num_epochs # moving average
    best_val_loss = 9999.0
    val_loss_MA = None
    best_val_loss_MA = 100.0

    history = []
    logger.info('starting training fold ({})...'.format(f))

    lr = lr_0
    lr_from_history = False

    load_start = time.time()

    tr_dataset = myH5PYDatasetSeg3D(s=s, c=c, phase='train', folds=(NCV, f), n_classes=n_classes)
    n_train = tr_dataset.len()

    # print('n_train = {}'.format(n_train))
    n_batch=1
    train_loader = DataLoader(tr_dataset,
                              batch_size = n_batch,
                              shuffle = True,
                              num_workers = 16,
                              pin_memory=True)

    if f == NCV:
        data_phase = 'train'
    else:
        data_phase = 'test'

    te_dataset = myH5PYDatasetSeg3D(s=s, c=c, phase=data_phase, folds=(NCV, f), n_classes=n_classes)

    n_test = te_dataset.len()

    test_loader = DataLoader(te_dataset,
                              batch_size = n_batch,
                              shuffle = False,
                              num_workers = 16,
                              pin_memory=True)

    load_end = time.time()
    logger.info('took {:.3f} seconds to load data: n_train = {}, n_test = {}'.format(load_end-load_start, n_train, n_test))

    # set up learning rate decay strategy, similar to lr_scheduler in PyTorch
    def set_lr2(epoch, prev_epoch, lr0, num_epochs): # add num_epochs on 3/23/21 by YY

        lr = lr0
        if epoch < max(150,num_epochs/2):
            lr = lr_0
        else:
            intval = np.clip(num_epochs/10, a_min=30, a_max=50)
            if (epoch - prev_epoch) >= intval:
                lr = max(0.3* lr0, 1e-6)

        assert(lr >= 1e-6)

        return lr

    def set_lr_cosine_warmup(epoch, lr0, num_epochs):
        lr_min = lr0 * 0.05
        epoch0 = 5
        if epoch < epoch0:
            lr = lr_min + (epoch-1) * (lr0 )/(epoch0-1) # [lr_min, lr_min + lr0]
        else:
            epoch_t = epoch - epoch0
            total_t = num_epochs - epoch0
            lr = 0.5 * (1 + np.cos(epoch_t * np.pi / total_t)) * lr0 + lr_min
        return lr

    def set_lr_cosine_annealing(epoch, lr0, num_epochs):

        epoch0 = 0
        lr_min = 1e-3 * lr0
        t_i = max(30.0, 0.1 * num_epochs)
        if epoch < epoch0:
            lr = lr0
        else:
            epoch_t = np.mod(epoch - epoch0, t_i)
            lr = lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(epoch_t * np.pi / t_i))
        return lr

    def poly_lr(epoch, max_epoch, lr0, n_warmup=0, power=0.9):

        lr_min = lr0 * 0.05
        if epoch < n_warmup:
            lr = lr_min + (1.0*epoch) * (lr0-lr_min) / (n_warmup - 1)  # [lr_min, lr0]
        else:
            lr = max(lr0 * (1-(1.0*(epoch-n_warmup))/max_epoch)**power, 1.0*lr0/max_epoch)
        return lr

    def set_optim(optimizer, lr):
        optim_str = 'adam'
        if lr < 0.00008:
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr,
                momentum=0.9,
                weight_decay=0
            )
            optim_str = 'sgd'
        return optimizer, optim_str
    prev_epoch = best_loss_epoch

    log_start = min(num_epochs * 0.2, 60)
    vidx = 1


    # model training
    print_stdout_to_file('\n---------- Start model training! ({}) ----------\n'.
                         format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), stdout_file)
    for epoch in range(start_epoch, num_epochs):
        # determine learning rate
        if lr_decay is 'step':
            lr0 = lr
            if lr_from_history == False:
                lr = set_lr2(epoch, prev_epoch, lr0, num_epochs=num_epochs)
            else:
                lr_from_history = False
                prev_epoch = epoch
            if lr != lr0:
                prev_epoch = epoch
        elif lr_decay is 'poly':
            lr = poly_lr(epoch, num_epochs, lr0=lr_0, n_warmup=0)
        elif lr_decay is 'cosine':
            lr = set_lr_cosine_warmup(epoch + 1, lr_0, num_epochs=num_epochs)
        elif lr_decay is 'cosine_annealing':
            lr = set_lr_cosine_annealing(epoch, lr_0, num_epochs=num_epochs)
        else:
            raise ValueError(lr_decay + ' is not implemented yet!')

        # if any of the following values is changed, we need to save the checkpoints
        best_loss_changed = False # best validation
        best_loss_MA_changed = False # best moving average of the validation

        print_stdout_to_file('', stdout_file)
        print_stdout_to_file('Epoch [{}/{}|{}] [v{}/f{}/g{}/vi{}/{}] (lr {:.6f}):'.
                             format(epoch, num_epochs - 1, prev_epoch, tag, f, args.gpu, vidx, optim_str, lr),
                             stdout_file)

        # train model
        start_time = time.time()
        train_metrics, prt_str = train(train_loader, net, loss, c, optimizer, lr=lr, amp_grad_scaler=amp_grad_scaler)
        # print_stdout_to_file(prt_str, stdout_file)
        train_loss = train_metrics[0]
        train_time = time.time() - start_time
        print_stdout_to_file(' - training loss ({:.2f} s):\t\t{:.4f}\t'.format(train_time, train_loss)
              + '\t'.join(['{:.4f}'.format(train_metrics[i]) for i in range(1, len(train_metrics))]),
                             stdout_file)
        # validate model
        start_time = time.time()
        valid_metrics, prt_str = validate(test_loader, net, val_loss, c, amp_grad_scaler=amp_grad_scaler)
        # print_stdout_to_file(prt_str, stdout_file)
        valid_loss = valid_metrics[vidx]
        valid_time = time.time() - start_time

        if val_loss_MA is None:
            val_loss_MA = valid_loss
        else:
            val_loss_MA = val_ma_alpha * val_loss_MA + (1.0 - val_ma_alpha) * valid_loss

        print_stdout_to_file(' - validing loss ({:.2f} s):\t\t{:.4f}\t{:.4f}|\t'.format(valid_time, val_loss_MA, valid_loss)
             + '\t'.join(['{:.4f}'.format(valid_metrics[i]) for i in range(0, len(valid_metrics))]),
                             stdout_file)

        # update training history
        history.append((train_loss, valid_loss, train_metrics, valid_metrics))

        if val_loss_MA <= best_val_loss_MA:
            best_loss_MA_changed = True
            best_loss_MA_epoch = epoch
            best_val_loss_MA = val_loss_MA

        if valid_loss < best_val_loss:
            best_loss_changed = True
            best_loss_epoch = epoch
            best_val_loss = valid_loss

        print_stdout_to_file(' - BEST(val_loss_MA)[{:d}]:\t\t{:.4f}'.format(best_loss_MA_epoch, best_val_loss_MA),
                             stdout_file)
        print_stdout_to_file(' - BEST(val_loss)[{:d}]:\t\t\t\t\t{:.4f}'.format(best_loss_epoch, best_val_loss),
                             stdout_file)

        if best_loss_MA_changed or best_loss_changed:
            prev_epoch = epoch

        # save checkpoints, note all these will be done in CPU
        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        #save best_loss
        if best_loss_changed:
            torch.save({
                'save_dir': save_dir,
                'save_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                'state_dict': state_dict,
                'args': args,
                'config': c,
                'setting': s},
                os.path.join(save_dir, 'best_val_loss_f{}.ckpt'.format(f))
            )
            if (epoch+1) > log_start:
                logger.info('-- best(tr, val_loss)[{:d}] = ({:.4f}, {:.4f}), lr = {:.6f}***'.format(
                    epoch, history[epoch][0], history[epoch][1], lr))

        if best_loss_MA_changed:
            torch.save({
                'save_dir': save_dir,
                'save_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                'state_dict': state_dict,
                'args': args,
                'config': c,
                'setting': s},
                os.path.join(save_dir, 'best_val_loss_ma_f{}.ckpt'.format(f))
            )
            if (epoch + 1) > log_start:
                logger.info(
                    '-- best(tr, val_loss_MA)[{:d}] = ({:.4f}, {:.4f}), lr = {:.6f}'.format(epoch, history[epoch][0],
                                                                                            best_val_loss_MA, lr))
        # save history at each epoch
        history_dict = {
            'history': history,
            'best_loss_epoch': best_loss_epoch,
            'best_val_loss': best_val_loss,
            'val_loss_MA': val_loss_MA,
            'best_loss_MA_epoch': best_loss_MA_epoch,
            'best_val_loss_MA': best_val_loss_MA,
            'save_dir': save_dir,
            'save_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'state_dict': state_dict,
            'args': args,
            'config': c,
            'setting': s,
            'lr': lr,
        }
        if amp_grad_scaler is not None:
            history_dict['amp_grad_scaler'] = amp_grad_scaler.state_dict()

        torch.save(history_dict, os.path.join(save_dir, 'history_f{}.ckpt'.format(f)))

        # if no update in 60 epochs and the lr is already small enough, the entire training is terminated
        if (epoch - prev_epoch) > 60 and lr == 1e-6:
            break

if __name__ == '__main__':
    main()
