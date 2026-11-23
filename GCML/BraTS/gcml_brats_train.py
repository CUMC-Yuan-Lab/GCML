import argparse, logging
import os, time, glob
import numpy as np

from tqdm import tqdm
from importlib import import_module

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from data.data import myFLDatasetSeg3D

from utils import (setgpu, count_parameters, set_logger, init_weights, set_seed, get_loss)

def grads_are_finite(net, loss_value, max_norm):
    """Clip the gradients of net (if max_norm is set) and return True if both the loss and the total gradient norm are
    finite, i.e. it is safe to take an optimizer step. A non-finite gradient must not reach the optimizer, as it would
    permanently corrupt the weights and Adam's moment estimates."""
    total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm if max_norm is not None else float('inf'))
    return bool(torch.isfinite(total_norm) and torch.isfinite(loss_value.detach()))

def state_is_finite(state_dict):
    return all(bool(torch.isfinite(v).all()) for v in state_dict.values() if v.is_floating_point())

def outputs_are_finite(output):
    """Must be checked before the loss: F.binary_cross_entropy in the focal loss triggers a fatal CUDA device-side
    assert (not a catchable error) when the predictions are NaN."""
    outs = output if isinstance(output, (list, tuple)) else [output]
    return all(bool(torch.isfinite(o).all()) for o in outs)

def check_finite_weights(net, where):
    if not state_is_finite(net.state_dict()):
        raise FloatingPointError('non-finite model weights detected: ' + where)

def train(data_loader, net, loss, c, optimizer, lr, gpu, amp_grad_scaler=None):

    net.train()

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    metrics = []
    n_skipped = 0  # steps skipped because of a non-finite loss or gradient

    output_prob = []
    clss = []
    weights = []
    # cid = 0 # case id
    for (data, target, weight) in tqdm(data_loader):

        cls = np.max(target.numpy(), axis=(2,3,4))[:,-1]

        # check for all-zero ch in target
        # all0_target_ch = []
        # for ch in range(target.shape[1]):
        #     if np.max(target.numpy()[:,ch,...]) == 0:
        #         all0_target_ch.append(ch)
        # if len(all0_target_ch) > 0:
        #     print('case{}, all-zerotarget ch{}'.format(cid, all0_target_ch))
        # cid += 1

        if c['weight_sample'] is None:
            weights.append(weight.numpy())

        data = data.cuda(device=gpu, non_blocking=True)
        target = target.cuda(device=gpu, non_blocking=True)
        weight = weight.cuda(device=gpu, non_blocking=True)
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

            if not outputs_are_finite(output):
                n_skipped += 1
                continue

            if isinstance(output, list) and c['dp'] is False:
                loss_output = loss(output[0], target, weight)
            else:
                loss_output = loss(output, target, weight)
            loss_output[0].backward()
            if not grads_are_finite(net, loss_output[0], c['grad_norm_clip']):
                n_skipped += 1
                continue
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

    if n_skipped > 0:
        print('WARNING: skipped {}/{} training steps with non-finite loss or gradient'.format(n_skipped, len(data_loader)))
    if len(metrics) == 0:
        raise FloatingPointError('all {} training steps had non-finite loss or gradient'.format(len(data_loader)))

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

    r_model = copy.deepcopy(net.state_dict())
    # r_optimizer = torch.optim.Adam(net.parameters())
    # r_optimizer.load_state_dict(optimizer.state_dict())

    # return r_optimizer, r_model, train_loss, prt_str
    return r_model, train_loss, prt_str

def train_dml(data_loader, net_gr, net_gs, loss, c, optimizer_gr, lr, gpu, dml_weight, ml_round, kl_type=2):

    #     KL_Loss = nn.KLDivLoss(reduction='mean')  # for entire KL, YY modified on 7/12/23
    KL_Loss = nn.KLDivLoss(reduction='none')  # for regional KL, so KL_loss has same dim as prediction and target

    net_gr.train()
    net_gs.train()

    for param_group in optimizer_gr.param_groups:
        param_group['lr'] = lr

    metrics = []
    kl_metrics = []
    n_skipped_gr = n_skipped_gs = 0  # steps skipped because of a non-finite loss or gradient
    net_gr = net_gr.to(device=gpu)
    net_gs = net_gs.to(device=gpu)

    # optimizer_to(optimizer_gr, gpu)

    optimizer_gs = torch.optim.Adam(net_gs.parameters())  # create optimizer for gs net
    optimizer_gs.load_state_dict(optimizers[gr].state_dict())  # copy state from gr optimizer

    for rd in range(ml_round):
        print('Start ML round: {}/{}'.format(rd+1, ml_round))
        for (data, target, weight) in tqdm(data_loader):
            data = data.to(device=gpu, non_blocking=True)
            target = target.to(device=gpu, non_blocking=True)
            weight = weight.to(device=gpu, non_blocking=True)

            target = target.type(torch.cuda.FloatTensor)
            weight = weight.type(torch.cuda.FloatTensor)

            output_gr = net_gr(data) # list of 4 of [1,3,128,128]
            output_gs = net_gs(data)

            if (c['prob_output'] is False) and (c['orl'] is True):
                assert (isinstance(output_gr, list))
                assert (isinstance(output_gs, list))
                for o_i in range(len(output_gr)):
                    output_gr[o_i] = F.sigmoid(output_gr[o_i]).clamp(1e-7, 1.0 - 1e-7)
                for o_i in range(len(output_gs)):
                    output_gs[o_i] = F.sigmoid(output_gs[o_i]).clamp(1e-7, 1.0 - 1e-7)
            # output is a probability of tumor (0,1)

            # a non-finite output of either model would break the KL and the loss (see outputs_are_finite)
            if not (outputs_are_finite(output_gr) and outputs_are_finite(output_gs)):
                n_skipped_gr += 1
                n_skipped_gs += 1
                continue

            # YY added on 12/17/23 to dynamically move the two models closer (if the target output is correct) or away (if the target output is wrong) from each other
            # it's done by multiplying the KL_Loss with a pred_w matrix where 1 indicates a correct prediction while -1 for incorrect prediction so minimizing a negative KLD is equivalent to maximize
            # the divergence
            # Note that pred_w should have the same dimension as the target (and output[0]), and be detached from the graph so it's not included in the backprop
            # JC modied on 1/3/24 to use on brats21

            kl_gr = kl_gs = 0
            n_ch = output_gr[0].shape[1] # n_ch = 3
            for ch in range(n_ch):

                output_gr_ch = torch.unsqueeze((output_gr[0][:, ch, ...]), dim=1) # [1,1,128,128,128]
                output_gs_ch = torch.unsqueeze((output_gs[0][:, ch, ...]), dim=1)

                gs_det = output_gs_ch.detach() # [1,1,128,128,128]
                gr_det = output_gr_ch.detach()

                gr_input = torch.cat([output_gr_ch, 1.0 - output_gr_ch], dim=1) # [1,2,128,128,128]
                gs_target = torch.cat([gs_det, 1.0 - gs_det], dim=1) # [1,2,128,128,128]

                gs_input = torch.cat([output_gs_ch, 1.0 - output_gs_ch], dim=1)
                gr_target = torch.cat([gr_det, 1.0 - gr_det], dim=1)

                target_ch = torch.unsqueeze(target[:, ch, ...], dim=1) #[1,1,128,128,128]
                # prediction_weight_matrix: correct prediction: w = 1, incorrect prediction: w = -1
                with torch.no_grad():
                    gs_pred_mask = torch.zeros_like(gs_det) # [1,1,128,128,128]
                    assert gs_pred_mask.shape == target_ch.shape, "gs_pred_mask.shape != target_ch.shape"
                    gs_pred_mask[gs_det > 0.5] = 1
                    # convert[True, False] to int: True = 1, False = 0
                    gs_pred_w = torch.eq(gs_pred_mask, target_ch).long()  # [1,1,128,128,128]
                    gs_pred_w = 2.0 * (gs_pred_w - 0.5)  # [-1, 1]
                    # print(torch.unique(gs_pred_w))

                    gr_pred_mask = torch.zeros_like(gr_det)
                    assert gr_pred_mask.shape == target_ch.shape, "gr_pred_mask.shape != target_ch.shape"
                    gr_pred_mask[gr_det > 0.5] = 1
                    gr_pred_w = torch.eq(gr_pred_mask, target_ch).long()
                    gr_pred_w = 2.0 * (gr_pred_w - 0.5)
                    # print(torch.unique(gr_pred_w))
                    # assert(0)

                if kl_type == 1 or kl_type == 2:
                    w_gr_kl_loss = gs_pred_w * KL_Loss(torch.log(gr_input), gs_target).sum(dim=1) # [1,1,128,128,128]
                    w_gs_kl_loss = gr_pred_w * KL_Loss(torch.log(gs_input), gr_target).sum(dim=1)

                if kl_type == 1:
                    # option 1: entire ROI GCML
                    kl_gr = kl_gr + torch.mean(w_gr_kl_loss)
                    kl_gs = kl_gs + torch.mean(w_gs_kl_loss)
                elif kl_type == 2:
                    # option 2: regional GCML
                    kl_gr = kl_gr + (torch.sum(w_gr_kl_loss * target_ch) + torch.sum(w_gr_kl_loss * gr_det)) / (
                                torch.sum(target_ch) + torch.sum(gr_det) +1)
                    kl_gs = kl_gs + (torch.sum(w_gs_kl_loss * target_ch) + torch.sum(w_gs_kl_loss * gs_det)) / (
                                torch.sum(target_ch) + torch.sum(gs_det) +1)
                elif kl_type == 3:
                    # option 3: entire ROI GML
                    kl_gr = kl_gr + torch.sum(KL_Loss(torch.log(gr_input), gs_target).sum(dim=1) * target_ch) / (
                                torch.sum(target_ch) + 1)
                    kl_gs = kl_gs + torch.sum(KL_Loss(torch.log(gs_input), gr_target).sum(dim=1) * target_ch) / (
                                torch.sum(target_ch) + 1)
                elif kl_type == 4:
                    # option 4: regional GML
                    kl_gr_ch = KL_Loss(torch.log(gr_input), gs_target).sum(dim=1)
                    kl_gs_ch = KL_Loss(torch.log(gs_input), gr_target).sum(dim=1)

                    kl_gr = kl_gr + (torch.sum(kl_gr_ch * target_ch) + torch.sum(kl_gr_ch * gr_det)) / (
                            torch.sum(target_ch) + torch.sum(gr_det) + 1)
                    kl_gs = kl_gs + (torch.sum(kl_gs_ch * target_ch) + torch.sum(kl_gs_ch * gs_det)) / (
                            torch.sum(target_ch) + torch.sum(gs_det) + 1)

            kl_gr = kl_gr / n_ch
            kl_gs = kl_gs / n_ch
            # print('kl_gr={}, kl_gs={}'.format(kl_gr,kl_gs))

            if isinstance(output_gr, list) and c['dp'] is False:
                loss_output_gr = (1 - dml_weight) * loss(output_gr[0], target, weight)[0] + dml_weight * kl_gr
            else:
                loss_output_gr = (1 - dml_weight) * loss(output_gr, target, weight)[0] + dml_weight * kl_gr

            if isinstance(output_gs, list) and c['dp'] is False:
                loss_output_gs = (1 - dml_weight) * loss(output_gs[0], target, weight)[0] + dml_weight * kl_gs
            else:
                loss_output_gs = (1 - dml_weight) * loss(output_gs, target, weight)[0] + dml_weight * kl_gs

            optimizer_gr.zero_grad()
            optimizer_gs.zero_grad()
            loss_output_gr.backward()
            loss_output_gs.backward()

            # each model steps only if its own loss and gradient are finite
            gr_ok = grads_are_finite(net_gr, loss_output_gr, c['grad_norm_clip'])
            gs_ok = grads_are_finite(net_gs, loss_output_gs, c['grad_norm_clip'])
            if gr_ok:
                optimizer_gr.step()
                metrics.append(loss_output_gr.item())
                kl_metrics.append(kl_gr.item())
            else:
                n_skipped_gr += 1
            if gs_ok:
                optimizer_gs.step()
            else:
                n_skipped_gs += 1

    n_steps = ml_round * len(data_loader)
    if n_skipped_gr > 0 or n_skipped_gs > 0:
        print('WARNING: skipped non-finite mutual-learning steps: receiver {}/{}, sender {}/{}'.format(
            n_skipped_gr, n_steps, n_skipped_gs, n_steps))

    if len(metrics) > 0:
        metrics = np.asarray(metrics, np.float32)
        train_loss = np.mean(metrics, axis=0)
        kl_metrics = np.asarray(kl_metrics, np.float32)
        kl_mean = np.mean(kl_metrics, axis=0)
        kl_0 = kl_metrics[0]
        kl_end = kl_metrics[-1]
        kl_loss = np.asarray([kl_mean, kl_0, kl_end], dtype=np.float32)
    else:
        print('WARNING: all receiver steps were skipped, the receiver model is not updated by mutual learning')
        train_loss = np.float32('nan')
        kl_mean = kl_0 = kl_end = np.float32('nan')
        kl_loss = np.asarray([kl_mean, kl_0, kl_end], dtype=np.float32)

    print('kl_mean:{:.5f} kl_0:{:.5f} kl_end:{:.5f}'.format(kl_mean, kl_0, kl_end))
    # if 'fs' in c['loss']:
    #     fs = metrics[:,-1]
    #     print('fs: min = {:.5f}, max = {:.5f}, median = {:.5f}, mean = {:.5f}'.format(fs.min(), fs.max(),
    #

    # TODO clean up GPU and move model/optimizer to CPU
    del optimizer_gs
    del net_gs
    torch.cuda.empty_cache()
    net_gr = net_gr.to('cpu')
    # optimizer_to(optimizer_gr, 'cpu')

    # return train_loss, kl_loss
    r_model = copy.deepcopy(net_gr.state_dict())

    # r_optimizer = torch.optim.Adam(net_gr.parameters())  # create optimizer for gs net
    # r_optimizer.load_state_dict(optimizer_gr.state_dict())  # copy state from gr optimizer
    # return r_optimizer, r_model, train_loss, kl_loss
    return r_model, train_loss, kl_loss

def validate(data_loader, net, loss, c, gpu, amp_grad_scaler=None):

    net.eval()
    metrics = []

    # TODO move single model from CPU to GPU
    net = net.to(device=gpu)
    loss = loss.to(device=gpu)

    output_prob = []
    clss = []
    with torch.no_grad():
        for i, (data, target, weight) in enumerate(data_loader):

            cls = np.max(target.numpy(), axis=(2, 3, 4))[:, -1]
            data = data.cuda(device=gpu)
            target = target.cuda(device=gpu)
            target = target.type(torch.cuda.FloatTensor)
            weight = weight.to(device=gpu)
            weight = weight.type(torch.cuda.FloatTensor)

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

    # TODO move single model from GPU to CPU
    net = net.to('cpu')
    loss = loss.to('cpu')

    return valid_loss, prt_str

def evaluate_gml(tr_loader, te_loader, net_gr, net_gs, losses_gr, c,
                                       val_losses_gr, gpu, vidx):
    # evaluate receiver and sender models
    rtr, prt_str = validate(tr_loader, net_gr, losses_gr, c,
                                gpu=gpu)  # lossavg, loss0, lossdp
    rva, prt_str = validate(te_loader, net_gr, val_losses_gr, c,
                                gpu=gpu)  # lossavg, loss0, lossdp
    str, prt_str = validate(tr_loader, net_gs, losses_gr, c,
                                gpu=gpu)  # lossavg, loss0, lossdp
    sva, prt_str = validate(te_loader, net_gs, val_losses_gr, c,
                                gpu=gpu)  # lossavg, loss0, lossdp
    return [rtr[vidx],rva[vidx],str[vidx],sva[vidx]]

def fedavg(models, weights):
    m_avg = copy.deepcopy(models[0])
    for key in m_avg.keys():
        # m_avg[key] = weights[0] * m_avg[key].cpu()
        m_avg[key] = weights[0] * torch.nan_to_num(m_avg[key].cpu(), nan=0.0)
        for i in range(1, len(models)):
            # m_avg[key] += weights[i] * models[i][key].cpu()
            m_avg[key] += weights[i] * torch.nan_to_num(models[i][key].cpu(), nan=0.0)
    return m_avg

def gmlavg(models, gidx, vloss):
    m_avg = copy.deepcopy(models[gidx[0]]) #gr
    for key in m_avg.keys():
        m_avg[key] = (((1 - vloss[0]) * models[gidx[0]][key].cpu() + (1 - vloss[1]) * models[gidx[1]][key].cpu())
                         / (2 - sum(vloss)))
    return m_avg

# TODO get_args
def get_args():

    # global args
    parser = argparse.ArgumentParser(description='PyTorch 3D Deep Planning')
    parser.add_argument('--config', '-c', default='config_v0', type=str,
                        help='configuration file for model training/testing, placed in configs/.')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run (default: 300)')
    parser.add_argument('-f', '--fold', type=int, metavar='F',
                        help='the fold for cross-validation when training models')
    parser.add_argument('--batch-size', '-b', default=1, type=int, metavar='N',
                        help='mini-batch size (default: 1)')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float, metavar='LR',
                        help='initial learning rate (default: 0.003')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W',
                        help='weight decay (default: 0)')
    parser.add_argument('--gpu', default='all', type=str, metavar='N',
                        help='the GPUs used (default: all)')
    parser.add_argument('--save_dir', type=str, metavar='SD',
                        help='directory to save trained models, relative to <output_dir>/fl-params')
    parser.add_argument('--output_dir', type=str, metavar='OD', default='/mnt/data/brats2021/outputs/tmp',
                        help='root directory of all outputs: models in <output_dir>/fl-params/, logs and GCML '
                             'records in <output_dir>/logs/ (default: /mnt/data/brats2021/outputs)')
    parser.add_argument('--log', '-l', action='store_true',
                        help='display and save debug information')
    parser.add_argument('--reset', '-r', action='store_true',
                        help='clear saved trained model and reset the training procedure')
    parser.add_argument('--seed', type=int, default=1234, help='random seed to use, default=1234')
    parser.add_argument('--switch-optim', action='store_true',
                        help='switch the optimization algorithms from adam to regular SGD')
    parser.add_argument('--valid-idx', default=0, type=int,
                        help='the index of loss value when performing validation (default: 0)')
    parser.add_argument('--setting', default='SETTINGS_tr', type=str,
                        help='the file specifying where data are read/saved (default: SETTINGS_tr), placed in settings/.')
    parser.add_argument('--dml_round', default=1, type=int,
                        help='number of mutual-learning round(s) per epoch (default: 1)')
    parser.add_argument('--dml_weight', default=0.5, type=float,
                        help='weight of KL loss in total (default: 0.5)')
    parser.add_argument('--eval_post', default=False, action='store_true',
                        help='whether to evaluate incoming model after GML (default: True)')
    parser.add_argument('--eval_pre', default=False, action='store_true',
                        help='whether to evaluate incoming model before GML (default: True)')
    parser.add_argument('--gmlavg', default=False, action='store_true',
                        help='whether to merge incoming and local models after GML (default: True)')
    parser.add_argument('--kl_type', default=2, type=int, choices=[0, 1, 2, 3, 4],
                        help='the type of KL loss function, 0=skipMutualLearning 1=entireGCML, 2=regionalGCML, 3=entireGML, 4=regionalGML (default: 2)')
    parser.add_argument('--dml_pair', default=2, type=int, choices=[1, 2, 3, 4, 5, 6],
                        help='number of site pair(s) per round, (default: 2)')


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # get global args from command line input
    args = get_args()

    # logs and GCML records are saved in <output_dir>/logs/
    output_dir = os.path.abspath(args.output_dir)
    hostpath = os.path.join(output_dir, 'logs') + '/'
    if not os.path.exists(hostpath):
        os.makedirs(hostpath)

    # load setting
    s = import_module('settings.' + args.setting)
    s = s.setting
    # model and output locations follow --output_dir instead of the settings file
    s['param_dir'] = os.path.join(output_dir, 'fl-params') + '/'
    s['output_dir'] = output_dir + '/'

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

    logger = set_logger(logger, log_file, log=args.log)
    logger.info('*** start {} (config: {}, setting: {}) ***'.format(__file__, args.config, args.setting))

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

    # set the path to save model weights
    save_dir = args.save_dir
    if args.save_dir is None:
        save_dir = os.path.join(s['param_dir'], args.config)
    else:
        save_dir = os.path.join(s['param_dir'], save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # log setting, config and run-time arg
    logger.info('SETTING')
    for k, v in s.items():
        logger.info(' - {} = {}'.format(k,v))

    logger.info('CONFIG ({})'.format(args.config))
    for k, v in c.items():
        logger.info(' - {} = {}'.format(k,v))

    logger.info('ARGS')
    for arg in vars(args):
        logger.info(' - {} = {}'.format(arg, getattr(args,arg)))

    if args.reset:
        print('training reset')
        for rf in glob.glob(os.path.join(save_dir, '*_f{}.*'.format(f))):
            os.remove(rf)
            print(' - removed ' + rf)

    # set up learning rate decay strategy, similar to lr_scheduler in PyTorch
    def set_lr2(epoch, prev_epoch, lr0, num_epochs): # add num_epochs on 3/23/21 by YY

        lr = lr0
        if epoch < max(150,num_epochs/2):
            lr = args.lr
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

        epoch0 = 0 #min(10, int(0.05*num_epochs))
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
                weight_decay=args.weight_decay
            )
            optim_str = 'sgd'
        return optimizer, optim_str

    # set up the epoch to start logging the training process - the purpose is not to log unuseful information as it's
    # very unlikely the best validation will take place in the first 20 - 30% of epochs
    log_start = min(args.epochs * 0.2, 60)

    # lr0 = lr

    # tracking validation loss and its moving average based on the $vidx-th loss when validation loss function includes
    # multiple loss components. default vidx = 0 - the total loss
    vidx = args.valid_idx

    # TODO from hecktor20
    # each client should have the following:
    # 1. fnames and data
    # 2. gpu
    # 3. model
    # 4. optimizer
    # 5. checkpoints: best_val, best_val_ma, last

    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    nodes = c['nodes']
    n_nodes = len(nodes)
    # assert (n_gpu == n_nodes)

    # load history if it exists and not over args.epochs
    num_epochs = args.epochs
    start_epoch = 0
    best_loss_epoch = n_nodes * [num_epochs] #num_epochs * np.ones((n_nodes, 1))
    best_loss_MA_epoch = n_nodes * [num_epochs] #num_epochs * np.ones((n_nodes, 1))  # moving average
    best_err_epoch = n_nodes * [num_epochs] #num_epochs * np.ones((n_nodes, 1)) # absolute difference
    prev_epoch = n_nodes * [num_epochs]

    best_val_loss = n_nodes * [100.0] #100.0 * np.ones((n_nodes, 1))
    best_val_err = n_nodes * [100.0] #100.0 * np.ones((n_nodes, 1))
    # add moving average on validation loss on 8/16/19
    val_ma_alpha = n_nodes * [0.0] #0.0 * np.ones((n_nodes, 1))
    if 'val_ma_alpha' in c.keys(): val_ma_alpha = c['val_ma_alpha']
    val_loss_MA = n_nodes * [None]
    best_val_loss_MA = n_nodes * [100.0] #100.0 * np.ones((n_nodes, 1))

    # ugly initialization (should be rewritten with OOP!)
    fnames = []
    nets = []
    gpus = []
    losses = []
    val_losses = []
    optimizers = []
    tr_loaders = []
    te_loaders = []
    lrs = []
    historys = []
    n_trains = []
    for i, node in enumerate(nodes):
        # data
        config_local = import_module('configs.' + 'config-site-' + node)
        lc = config_local.config
        if c['prep_ver'] in [1, 2]:
            fname = lc['fname_list']
        else:
            raise ValueError('{} is not implemented yet'.format(c['prep_ver']))
        fname = os.path.join('configs', fname)
        logger.info('file list: ' + fname)

        # TODO init model, assign gpu
        # create DL model and initialize weights
        model = import_module('models.' + model_name)
        net = model.Net(in_channels=in_channels, nclasses=out_channels, nf=nf, relu=nonlin, up_mode=up_mode,
                        dropout_p=dropout_p,
                        depth=depth, padding=True, group_norm=c['group_norm'], deep_supervision=c['dp'])

        set_seed(args.seed)
        init_weights(net, init_type=init, nonlin=nonlin_w)

        n_params = count_parameters(net)
        logger.info(' - param_count = {0:,}'.format(n_params))

        # set GPU and move network model and loss functions to GPU
        loss = get_loss(c['loss'], dp=dp)

        gpu = torch.device('cuda:{}'.format(i % n_gpu))
        # net = net.to(device=gpu)
        # loss = loss.to(device=gpu)
        net = net.to('cpu')
        loss = loss.to('cpu')


        # val_loss = loss if 'val_loss' is not specified in configuration
        if 'val_loss' not in c.keys():
            c['val_loss'] = c['loss']
            val_loss = loss
        else:
            val_loss = get_loss(c['val_loss'], dp=dp)
            val_loss = val_loss.cuda()

        vloss_str = c['val_loss']

        # data parallel is used to distribute data to train a model on multiple GPUs (when batch_size > 1 and n_gpu > 1)
        # net = DataParallel(net)

        # set optimization methods
        if c['optim'] == 'sgd':  # default settings from nnunet 07/28/21
            optimizer = torch.optim.SGD(
                net.parameters(),
                args.lr,
                momentum=0.99,
                weight_decay=args.weight_decay,
                dampening=0,
                nesterov=True
            )
        elif c['optim'] == 'adam':
            optimizer = torch.optim.Adam(
                net.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                amsgrad=True
            )
        elif c['optim'] == 'adamw':
            optimizer = torch.optim.AdamW(
                net.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                amsgrad=True
            )
        else:
            logger.info(c['optim'] + ' is not implemented yet')
            raise ValueError

        optim_str = c['optim']

        # TODO load history
        history = []
        hist_file = os.path.join(save_dir, node + '_history_f{}.ckpt'.format(f))
        print('search model training history from : ' + hist_file)

        if os.path.isfile(hist_file):
            checkpoint = torch.load(hist_file)
            history = checkpoint['history']
            best_loss_epoch[i] = checkpoint['best_loss_epoch']
            best_val_loss[i] = history[best_loss_epoch[i]][1]

            if 'lr' in checkpoint.keys():
                lr = checkpoint['lr']
            if 'val_loss_MA' in checkpoint.keys():
                val_loss_MA[i] = checkpoint['val_loss_MA']
            if 'best_val_loss_MA' in checkpoint.keys():
                best_val_loss_MA[i] = checkpoint['best_val_loss_MA']
            else:
                best_val_loss_MA[i] = checkpoint['best_val_loss']
            if 'best_loss_MA_epoch' in checkpoint.keys():
                best_loss_MA_epoch[i] = checkpoint['best_loss_MA_epoch']
            else:
                best_loss_MA_epoch[i] = best_loss_epoch[i]

            start_epoch = len(history)
            if start_epoch < num_epochs:
                # net.load_state_dict(checkpoint['state_dict'])
                net.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info(node + ':continue the previous training from epoch ({:d}), lr = {}'.format(start_epoch, lr))
                t = best_loss_epoch[i]
                logger.info(node + ': (tr_loss, BEST(val_loss)[{:d}] = ({:.6f}, *{:.6f})'.
                            format(t, history[t][0], best_val_loss[i]))
                logger.info(node + ': (tr_loss, BEST(val_loss_MA)[{:d}] = ({:.6f}, *{:.6f})'.
                            format(best_loss_MA_epoch[i], history[best_loss_MA_epoch[i]][0], best_val_loss_MA[i]))
                lr_from_history = True
                if c['amp']:
                    if 'amp_grad_scaler' in checkpoint.keys() and amp_grad_scaler is not None:
                        amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])
            else:
                logger.info(node + ': reached the maximum epochs ({:d})'.format(start_epoch))
                logger.info(node + ': BEST(train_loss, val_loss)[{:d}] = ({:.6f}, {:.6f})'.
                      format(best_loss_epoch[i], history[best_loss_epoch[i]][0], best_val_loss[i]))
                exit()
        else:
            if c['pretrained'] is not None:
                #pretrained_file = os.path.join(s['param_dir'], c['pretrained'] + '_f{}.ckpt'.format(f))
                pretrained_file = os.path.join(s['param_dir'], c['pretrained'])
                pretrained_checkpoint = torch.load(pretrained_file)
                net.load_state_dict(pretrained_checkpoint['state_dict'])
                logger.info(node + ': load pretrained model from ' + pretrained_file)

        lr = args.lr
        lr_from_history = False
        # lr_decay_strategy
        lr_decay = 'step'
        if 'lr_decay' in c.keys():
            lr_decay = c['lr_decay']

        # TODO load data
        load_start = time.time()

        c['fname_list']= fname
        c['index_train']= lc['index_train']
        c['index_valid']= lc['index_valid']

        tr_dataset = myFLDatasetSeg3D(s=s, c=c, phase='train', folds=(NCV, f), n_classes=n_classes)
        n_train = tr_dataset.len()

        # print('n_train = {}'.format(n_train))
        train_loader = DataLoader(tr_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=8, #turn off to avoid pydevd_attach_to_process error
                                  pin_memory=True)


        te_dataset = myFLDatasetSeg3D(s=s, c=c, phase='test', folds=(NCV, f), n_classes=n_classes)

        n_test = te_dataset.len()

        test_loader = DataLoader(te_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=8,  #turn off to avoid pydevd_attach_to_process error
                                 pin_memory=True)

        load_end = time.time()
        logger.info('took {:.3f} seconds to load data: n_train = {}, n_test = {}'.format(load_end - load_start, n_train,
                                                                                         n_test))

        gpus.append(gpu)
        nets.append(net)
        losses.append(loss)
        val_losses.append(val_loss)
        tr_loaders.append(train_loader)
        te_loaders.append(test_loader)
        lrs.append(lr)
        optimizers.append(optimizer)
        historys.append(history)
        n_trains.append(n_train)

    # TODO train
    n_trains = np.asarray(n_trains, dtype=np.float32)
    n_weights = n_trains / n_trains.sum()
    print('n_trains = {}, n_weights = {}'.format(n_trains, n_weights))
    # assert(0)
    # print('lengths: ', len(tr_loaders), len(nets), len(losses), len(optimizers), len(lrs), len(gpus))
    log_start = min(args.epochs * 0.2, 60)

    local_models = n_nodes * [None]
    
    n_warmup = 0
    if 'n_warmup' in c.keys():
        n_warmup = c['n_warmup']
    logger.info(' - n_warmup = {}'.format(n_warmup))

    # GCML (gossip learning) args
    Eval = args.eval_post #True  # post-GML model performance evaluations
    Eval_pre = args.eval_pre #True  # pre-GML model performance evaluations
    GML_avg = args.gmlavg #True # JC added 9/3/23, GMLavg, requires Eval=Ture
    if GML_avg == True:
        assert Eval == True, 'eval_post must be turned on to allow gml_avg'

    # default dml_weight fixed to 0.5
    dml_weight = args.dml_weight #0.9  # 0.5
    ML_round = args.dml_round #1
    KL_type = args.kl_type #2
    DML_pair = args.dml_pair  #2
    assert DML_pair*2 <= n_nodes, 'DML_pair must be less than or equal to half of n_nodes'

    gl_history = np.zeros(((args.epochs - n_warmup)*DML_pair, 4), dtype=np.int) #[ep, dpr, gr, gs]
    kl_history = np.zeros(((args.epochs - n_warmup)*DML_pair, 5), dtype=np.float32) #[ep, dpr, [kl_mean, kl_0, kl_end]]
    if Eval_pre == True:
        pre_eval = np.zeros(((args.epochs - n_warmup)*DML_pair, 6), dtype=np.float32) #[ep, dpr, rtr, rva, str, sva]
    if Eval == True:
        post_eval = np.zeros(((args.epochs - n_warmup)*DML_pair, 6), dtype=np.float32) #[ep, dpr, rtr, rva, str, sva]

    enable_m_weights = False  # m_weight not needed
    if 'enable_m_weights' in c.keys():
        enable_m_weights = c['enable_m_weights'] # used to select sender
    m_weights = np.copy(n_weights)
    # assert(0)
    for epoch in range(start_epoch, args.epochs):
        for i, node in enumerate(nodes):
            lr_from_history = False
            lr0 = lr = lrs[i]
            prev_epoch0 = prev_epoch[i]
            if lr_from_history == False:
                if lr_decay == 'step':
                    # lr = set_lr2(epoch, prev_epoch0, lr0)
                    lr = set_lr2(epoch, prev_epoch0, lr0, num_epochs=args.epochs)
                elif lr_decay == 'cosine':
                    lr = set_lr_cosine_warmup(epoch+1, args.lr, num_epochs=args.epochs)
                else:
                    raise ValueError(lr_decay + ' is not implemented yet!')
            else:
                lr_from_history = False
                prev_epoch[i] = epoch

            if lr != lr0:
                prev_epoch[i] = epoch

            best_loss_changed = False
            best_loss_MA_changed = False
            best_err_changed = False

            print('')
            print('NODE: {} [{}/{}]'.format(node, i+1, n_nodes))
            print('Epoch [{}/{}|{}] [v{}/{}/f{}/g{}/vi{}/{}] (lr {:.6f}):'.format(epoch, args.epochs - 1,prev_epoch[i], tag,
                                                                                  node, f, i, vidx, optim_str, lr))
            start_time = time.time()
            # print('i',i)
            # TODO move model from CPU to GPU
            nets[i] = nets[i].to(device=gpus[i])
            losses[i] = losses[i].to(device=gpus[i])
            # optimizer_to(optimizers[i], gpus[i])

            # TODO local train
            r_model, train_metrics, prt_str = train(tr_loaders[i], nets[i], losses[i], c, optimizers[i], lr=lr, gpu=gpus[i])  # lossavg, loss0, lossdp
            nets[i].load_state_dict(r_model)
            # optimizers[i].load_state_dict(r_optimizer.state_dict())
            check_finite_weights(nets[i], 'node {}, epoch {}, after local training'.format(node, epoch))

            train_loss = train_metrics[vidx]  #[0] export loss0 for fair compare with valid_loss
            train_time = time.time() - start_time

            print(' - training loss ({:.2f} s):\t\t{:.8f}\t'.format(train_time, train_loss)
              + '\t'.join(['{:.8f}'.format(train_metrics[i]) for i in range(1, len(train_metrics))]))

            start_time = time.time()
            valid_metrics, prt_str = validate(te_loaders[i], nets[i], val_losses[i], c, gpu=gpus[i])  # lossavg, loss0, lossdp
            valid_loss = valid_metrics[vidx]
            valid_time = time.time() - start_time
            if val_loss_MA[i] is None:
                val_loss_MA[i] = valid_loss
            else:
                val_loss_MA[i] = val_ma_alpha * val_loss_MA[i] + (1.0 - val_ma_alpha) * valid_loss

            print(' - validing loss ({:.2f} s):\t\t{:.8f}\t{:.8f}|\t'.format(valid_time, val_loss_MA[i], valid_loss)
             + '\t'.join(['{:.8f}'.format(valid_metrics[i]) for i in range(0, len(valid_metrics))]))

            historys[i].append((train_loss, valid_loss))

            # plot train_loss, valid_loss over epoch from historys[i]: (train_metrics[0], valid_metrics[vidx])
            # plot_loss(historys[i], (c['prefix'] + "_" + node))
            # print(node + ": loss plot updated")

            if val_loss_MA[i] <= best_val_loss_MA[i]:
                best_loss_MA_changed = True
                best_loss_MA_epoch[i] = epoch
                #best_val_loss = valid_loss
                best_val_loss_MA[i] = val_loss_MA[i]
                prev_epoch[i] = epoch
            if valid_loss < best_val_loss[i]:
                best_loss_changed = True
                best_loss_epoch[i] = epoch
                best_val_loss[i] = valid_loss
                prev_epoch[i] = epoch

            print(' - BEST(val_loss_MA)[{:d}]:\t\t{:.8f}'.format(best_loss_MA_epoch[i], best_val_loss_MA[i]))
            print(' - BEST(val_loss)[{:d}]:\t\t\t\t\t{:.8f}'.format(best_loss_epoch[i], best_val_loss[i]))

            # only the best_val_loss model of each site is saved, whenever its validation loss reaches a new minimum
            if best_loss_changed:
                state_dict = nets[i].state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].cpu()

                torch.save({
                    'save_dir': save_dir,
                    'save_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    'state_dict': state_dict,
                    'args': args,
                    'config': c,
                    'setting': s},
                    os.path.join(save_dir, node + '_best_val_loss_f{}.ckpt'.format(f))
                )
                if (epoch+1) > log_start:
                    logger.info(node + ': best(tr, val_loss)[{:d}] = ({:.6f}, {:.6f}), lr = {:.6f}'.format(epoch, historys[i][epoch][0],
                                                                                   historys[i][epoch][1], lr))

            # TODO move model from GPU to CPU
            nets[i] = nets[i].to('cpu')
            losses[i] = losses[i].to('cpu')
            # optimizer_to(optimizers[i], 'cpu')

            # copy weights from local models to cpu
            local_models[i] = copy.deepcopy(nets[i].state_dict())
            print(node + ': local model is ready for GFL!')

        # assert(0)
        if epoch >= n_warmup:
            # TODO gl
            rs = np.asarray([r for r in range(n_nodes)])  # initial pool
            for dpr in range(DML_pair):
                # print('m_weights = {}'.format(m_weights))
                rs_weights = np.asarray([m_weights[r] for r in rs]) # weights of rs
                rs_weights = rs_weights / rs_weights.sum() # normalize
                gs = np.random.choice(rs, size=1, replace=False, p=rs_weights).astype(int)[0]  # select sender
                rs = np.asarray([r for r in rs if r != gs])  # update pool
                gr = np.random.choice(rs, size=1).astype(int)[0]  # select receiver
                rs = np.asarray([r for r in rs if r != gr])  # update pool
                print('Pair {}/{}: rs={}, gr={}, gs={}'.format(dpr+1, DML_pair, rs, gr, gs))
                gidx = np.asarray([gr, gs], dtype=np.int)
                assert((len(gidx) ==2) and (gidx[0] != gidx[1]))

                # mutual-learning with KLDivLoss
                # transfer model from gs to gr
                print('site {} received model from site {} for GML type{}'.format(nodes[gr], nodes[gs], KL_type))
                net_gs = model.Net(in_channels=in_channels, nclasses=out_channels, nf=nf, up_mode=up_mode,
                                dropout_p=dropout_p, group_norm=gn, depth=depth, padding=True, deep_supervision=dp)  # get new instance
                net_gs.load_state_dict(nets[gs].state_dict())  # copy state

                idpr =(epoch - n_warmup) * DML_pair + dpr # dpr index

                net_gs = net_gs.to(device=gpus[gr])
                # TODO pre-gml evaluation
                if Eval_pre == True:
                    pre_eval[idpr][0:2] = [epoch, dpr]
                    pre_eval[idpr][2:6] = evaluate_gml(tr_loaders[gr], te_loaders[gr], nets[gr], net_gs, losses[gr], c,
                                           val_losses[gr], gpu=gpus[gr], vidx=vidx)  # [gr_tr,gr_va,gs_tr,gs_va]
                    print('Pre-GML: r={}, s={}, rTr={:.4f}, rVa={:.4f}, sTr={:.4f}, sVa={:.4f}'.format(nodes[gr],nodes[gs],
                            pre_eval[idpr][2],pre_eval[idpr][3],pre_eval[idpr][4],pre_eval[idpr][5]))

                print('gidx = {}, dml_weight = {}'.format(gidx, dml_weight))

                # TODO GML
                if KL_type != 0: # KL_type = 0 to skip contrastive mutual learning
                    # train_metrics, kl_loss = train_dml(tr_loaders[gr], nets[gr], net_gs, losses[gr], c, optimizers[gr],
                    #                           lr=lr, gpu=gpus[gr], dml_weight=dml_weight)
                    r_model, train_metrics, kl_loss = train_dml(tr_loaders[gr], nets[gr], net_gs, losses[gr], c, optimizers[gr],
                                            lr=lr, gpu=gpus[gr], dml_weight=dml_weight, ml_round=ML_round, kl_type=KL_type)
                    nets[gr].load_state_dict(r_model)

                # post-gml evaluation
                if Eval == True:
                    post_eval[(epoch - n_warmup) * DML_pair + dpr][0:2] = [epoch, dpr]
                    post_eval[idpr][2:6] = evaluate_gml(tr_loaders[gr], te_loaders[gr], nets[gr], net_gs, losses[gr], c,
                                           val_losses[gr], gpu=gpus[gr], vidx=vidx) # [gr_tr,gr_va,gs_tr,gs_va]
                    print('Post-GML: r={}, s={}, rTr={:.4f}, rVa={:.4f}, sTr={:.4f}, sVa={:.4f}'.format(nodes[gr],nodes[gs],
                            post_eval[idpr][2],post_eval[idpr][3],post_eval[idpr][4],post_eval[idpr][5]))

                # TODO GMLavg
                if GML_avg == True:
                    print('Performing GMLavg...')
                    assert(Eval == True)
                    pgvl = [post_eval[idpr][i] for i in [3,5]] # [gr_va, gs_va]
                    # print('pgvl = {}'.format(pgvl))
                    # print('post_eval = {}'.format(post_eval[idpr]))

                    # update local models
                    local_models[gr] = copy.deepcopy(nets[gr].state_dict())
                    local_models[gs] = copy.deepcopy(net_gs.state_dict())

                    # skip the merge if either model has non-finite weights or a validation loss is not usable, as
                    # NaN/inf would propagate to the receiver even with a zero weight
                    if (np.all(np.isfinite(pgvl)) and sum(pgvl) < 2.0 and state_is_finite(local_models[gr])
                            and state_is_finite(local_models[gs])):
                        gmlavg_model = gmlavg(local_models, gidx, pgvl)
                        nets[gr].load_state_dict(gmlavg_model)
                    else:
                        print('WARNING: GMLavg skipped at epoch {} for receiver {} and sender {} (non-finite weights or '
                              'validation loss, pgvl = {}); the receiver keeps its post-GML model'.format(
                                epoch, nodes[gr], nodes[gs], pgvl))

                check_finite_weights(nets[gr], 'node {}, epoch {}, after mutual learning with node {}'.format(
                    nodes[gr], epoch, nodes[gs]))

                del net_gs
                torch.cuda.empty_cache()

                if KL_type != 0:
                    kl_history[idpr][0:2] = [epoch, dpr]
                    kl_history[idpr][2:5] = kl_loss

                gl_history[idpr][0:2] = [epoch, dpr]
                gl_history[idpr][2:5] = gidx

                if enable_m_weights:
                    print('before re-weight m_weights:', m_weights)
                    m_weights[gs] = m_weights[gs] / 2.0
                    m_weights[gr] = m_weights[gr] + m_weights[gs]
                    print('before after-weight m_weights:', m_weights)
                    print('normalized m_weight:', m_weights / m_weights.sum())

        # # TODO wait for cpu to cool down
        # print('waiting for cpu to cool down...')
        # time.sleep(60)


    # TODO valid
    # final validation
    for i, node in enumerate(nodes):
        valid_checkpoint = torch.load(os.path.join(save_dir, node + '_best_val_loss_f{}.ckpt'.format(f)))
        nets[i].load_state_dict(valid_checkpoint['state_dict'])
        valid_metrics, prt_str = validate(te_loaders[i], nets[i], val_losses[i], c, gpu=gpus[i])
        valid_loss = valid_metrics[vidx]
        # update local_models
        local_models[i] = copy.deepcopy(valid_checkpoint['state_dict'])

        logger.info(node + ': BEST(val_loss)[{:d}]):  {:.8f} | '.format(best_loss_epoch[i], valid_loss)
                    + ' '.join(['{:.8f}'.format(valid_metrics[i]) for i in range(0, len(valid_metrics))]))

    # model transaction statistics for gossip learning
    # print(gl_history)
    logger.info('Gossip Learning summary')
    # create an aggregated GL model from the best_val_loss models (evaluated below, not saved)
    gl_model = fedavg(local_models, n_weights)


    glh_file = hostpath + c['prefix'] + '_tr_v{}_f{}_gl-history.csv'.format(c['tag'], f)
    np.savetxt(glh_file, gl_history, delimiter=',')
    logger.info('gl_history is saved as ' + glh_file)

    if KL_type != 0:
        klh_file = hostpath + c['prefix'] + '_tr_v{}_f{}_kl-history.csv'.format(c['tag'], f)
        np.savetxt(klh_file, kl_history, delimiter=',')
        logger.info('kl_history is saved as ' + klh_file)

    if Eval_pre == True:
        pre_file = hostpath + c['prefix'] + '_tr_v{}_f{}_pre_eval.csv'.format(c['tag'], f)
        np.savetxt(pre_file, pre_eval, delimiter=',')
        logger.info('pre-gml evaluation is saved as ' + pre_file)

    if Eval == True:
        post_file = hostpath + c['prefix'] + '_tr_v{}_f{}_post_eval.csv'.format(c['tag'], f)
        np.savetxt(post_file, post_eval, delimiter=',')
        logger.info('post-gml evaluation is saved as ' + post_file)

    for i, node in enumerate(nodes):
        gl_i = gl_history[np.where(gl_history[:, 2] == i)]  # [ep, dpr, gr, gs]
        for j, node2 in enumerate(nodes):
            if i != j:
                # n_itoj = 0
                n_jtoi = gl_i[np.where(gl_i[:, 3] == j)].shape[0]
                logger.info(' {} <-- {}: {}'.format(node, node2, n_jtoi))

        # validate using gl_model
        nets[i].load_state_dict(gl_model)
        # gl_checkpoint = torch.load(os.path.join((save_dir, 'gl_best_val_loss_f0.ckpt')))
        # nets[i].load_state_dict(gl_checkpoint['state_dict'])
        valid_metrics, prt_str = validate(te_loaders[i], nets[i], val_losses[i], c, gpu=gpus[i])
        valid_loss = valid_metrics[vidx]
        logger.info(node + ': GL(val_loss)[{:d}]):  {:.8f} | '.format(args.epochs, valid_loss)
                    + ' '.join(['{:.8f}'.format(valid_metrics[i]) for i in range(0, len(valid_metrics))]))