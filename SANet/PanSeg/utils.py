import os, logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('agg')

from torch.cuda.amp import autocast

def getFreeId():
    import pynvml

    pynvml.nvmlInit()
    def getFreeRatio(id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5*(float(use.gpu+float(use.memory)))
        return ratio

    deviceCount = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(deviceCount):
        if getFreeRatio(i)<70:
            available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus+str(g)+','
    gpus = gpus[:-1]
    return gpus

def setgpu(gpuinput):
    freeids = getFreeId()
    if gpuinput=='all':
        gpus = freeids
    else:
        gpus = gpuinput
        if any([g not in freeids for g in gpus.split(',')]):
            raise ValueError('gpu'+g+'is being used')
    print('using gpu '+gpus)
    os.environ['CUDA_VISIBLE_DEVICES']=gpus
    return len(gpus.split(','))

def count_parameters(model):
    return (sum(p.numel() for p in model.parameters() if p.requires_grad))


def set_logger(logger, log_file, log=False):
    # assert(mode == 'debug' or mode == 'log')

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    if log:
        # print('display and save debug info')
        handler.setLevel(logging.INFO)
    else:
        # print('display debug info only, no log saved')
        handler.setLevel(logging.ERROR)

    logger.addHandler(handler)

    return logger

def weights_init_kaiming(m, nonlin='leaky_relu'):
    assert(nonlin=='relu' or nonlin == 'leaky_relu')
    # print('kaiming: uniform')
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        m.weight = nn.init.kaiming_normal_(m.weight, a=1e-2, mode='fan_in', nonlinearity=nonlin)
        if m.bias is not None:
            m.bias = nn.init.constant_(m.bias, 0)

def init_weights(net, init_type='kaiming', nonlin='relu'):
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError(
            'initialization method [{}] is not implemented'.format(init_type))


def print_stdout_to_file(txt, fname, mode='a'):
    with open(fname, mode) as stdout_:
        stdout_.write(txt+'\n')
    print(txt)

def sliding_window_inference(data, voi, net, patch_size, ncls, step_ratio=0.5, importance_map=1.0,
                             nonlin=lambda x:x, ckpt_list=None, amp=False, tta=False, verbose=True, gpu_swap=False):

    assert len(data.shape) == 4, "data must be (size_c, size_z,size_y, size_x)"
    step = patch_size * step_ratio
    zyx0 = [] # starting points of crop_voi

    for dim in range(3):
        zyx0.append(np.arange(start=voi[dim], stop=voi[dim+3]+voi[dim]-step[dim], step=step[dim], dtype=np.int16))
        if zyx0[dim][-1] + patch_size[dim] > voi[dim+3] + voi[dim]:
            zyx0[dim][-1] = voi[dim+3] + voi[dim] - patch_size[dim] # - 1
            # YY added below on 10/28/23 to evenly overlap ROIs with new step_size
            zyx0_diff = np.diff(zyx0[dim])
            new_step_size = int(np.floor(zyx0_diff.sum() / len(zyx0_diff)))
            for tt in range(1, len(zyx0[dim])-1):
                zyx0[dim][tt] = tt * new_step_size

    #print('zyx0: ',zyx0)
    # zyx0[0] = np.asarray([0,35,70,106])
    zyx = np.array(np.meshgrid(zyx0[0], zyx0[1], zyx0[2])).T.reshape(-1, 3).astype(np.int16)

    if verbose:
        print('data.shape: ', data.shape)
        print('voi: ', voi)
        print('step: ', step)
        print('ncls: ', ncls)
        print('n_zyx: ', zyx.shape[0])
        print('zyx0: ', zyx0)
    # print('n_zyx: ', zyx.shape[0])
    # assert(0)
    voi_output = []
    if ckpt_list is not None:
        i = 1
        for check_point in ckpt_list:
            net.load_state_dict(check_point)
            print(' loaded No. {} check_point'.format(i))
            if gpu_swap == False: # for pool and indv valid
                patch_output = patch_inference(data, voi, net, patch_size,zyx, ncls, importance_map, nonlin=nonlin, amp=amp,
                                            tta=tta, verbose=verbose, gpu_swap=False)
            else: # for all FL valid
                patch_output = patch_inference(data, voi, net, patch_size,zyx, ncls, importance_map, nonlin=nonlin, amp=amp,
                                            tta=tta, verbose=verbose, gpu_swap=True)

            voi_output.append(patch_output.cpu().numpy())

            i = i + 1

        voi_output = np.asarray(voi_output).astype(np.float32)
        voi_output = np.mean(voi_output, axis=0) # return np.array
    else:
        # return a tensor in gpu (for validation use)
        if gpu_swap == False:
            voi_output = patch_inference(data, voi, net, patch_size, zyx, ncls, importance_map,nonlin=nonlin, amp=amp,
                                         tta=tta, verbose=verbose, gpu_swap=False)
        else:
            voi_output = patch_inference(data, voi, net, patch_size, zyx, ncls, importance_map,nonlin=nonlin, amp=amp,
                                         tta=tta, verbose=verbose, gpu_swap=True)

    return voi_output

def sliding_window_inference_pred(data, voi, net, patch_size, ncls, step_ratio=0.5, importance_map=1.0,
                             nonlin=lambda x:x, ckpt_list=None, amp=False, tta=False, verbose=True):
    pad_z_roi = False
    if data.shape[1] < patch_size[0]: # z roi size is less than patch size
        pad_z_roi = True
        pad_z = patch_size[0] - data.shape[1]
        # pad tensor data to make z roi size equal to patch size
        data = torch.cat((data, torch.zeros(data.shape[0], pad_z, data.shape[2], data.shape[3], dtype=data.dtype)), dim=1)
        voi[3] = data.shape[1]
    # end of modification 1

    assert len(data.shape) == 4, "data must be (size_c, size_z,size_y, size_x)"
    step = patch_size * step_ratio
    zyx0 = [] # starting points of crop_voi

    for dim in range(3):
        zyx0.append(np.arange(start=voi[dim], stop=voi[dim+3]+voi[dim]-step[dim], step=step[dim], dtype=np.int16))
        if zyx0[dim][-1] + patch_size[dim] > voi[dim+3] + voi[dim]:
            zyx0[dim][-1] = voi[dim+3] + voi[dim] - patch_size[dim] # - 1
            zyx0_diff = np.diff(zyx0[dim])
            new_step_size = int(np.floor(zyx0_diff.sum() / len(zyx0_diff)))
            for tt in range(1, len(zyx0[dim])-1):
                zyx0[dim][tt] = tt * new_step_size


    zyx = np.array(np.meshgrid(zyx0[0], zyx0[1], zyx0[2])).T.reshape(-1, 3).astype(np.int16)

    if verbose:
        print('data.shape: ', data.shape)
        print('voi: ', voi)
        print('step: ', step)
        print('ncls: ', ncls)
        print('n_zyx: ', zyx.shape[0])
        print('zyx0: ', zyx0)
    voi_output = []
    if ckpt_list is not None:
        i = 1
        for check_point in ckpt_list:
            net.load_state_dict(check_point)
            print(' loaded No. {} check_point'.format(i))
            patch_output = patch_inference(data, voi, net, patch_size,zyx, ncls, importance_map, nonlin=nonlin, amp=amp,
                                           tta=tta, verbose=verbose, gpu_swap=False)
            voi_output.append(patch_output.cpu().numpy())

            i = i + 1

        voi_output = np.asarray(voi_output).astype(np.float32)
        voi_output = np.mean(voi_output, axis=0) # return np.array
    else:
        # return a tensor in gpu (for validation use)
        voi_output = patch_inference(data, voi, net, patch_size, zyx, ncls, importance_map,nonlin=nonlin, amp=amp,
                                     tta=tta, verbose=verbose)

    if pad_z_roi:
        voi_output = voi_output[:, :patch_size[0]- pad_z, :, :]

    return voi_output

def model_forward_pass(data, net, amp):
    if amp:
        with autocast():
            output = net(data)
    else:
        output = net(data)

    if isinstance(output, list): output = output[0]
    return output

def patch_inference(data, voi, net, patch_size, zyx, ncls, importance_map=1.0, nonlin=lambda x:x, amp=False, tta=False,
                    verbose=True, gpu_swap=True):
    if gpu_swap == False:
        output = torch.zeros((ncls, voi[3], voi[4], voi[5]), dtype=torch.float32, requires_grad=False).cuda(non_blocking=True)
        n_predictions = torch.zeros_like(output, dtype=torch.half, requires_grad=False).cuda(non_blocking=True) # float16

    pz, py, px = patch_size
    if verbose: print('output.shape ', output.shape, ', patch_size: ', patch_size)
    # print(data.device)
    # assert(0)
    if tta:
        mirror_idx = 8
        mirror_axes = (0,1,2)
        num_mirrors = 2 ** len(mirror_axes)
    else:
        mirror_idx = 1
        num_mirrors = 1

    for k, (z, y, x) in enumerate(zyx):
        data_k = torch.unsqueeze(data[:,z:z+pz, y:y+py, x:x+px], dim=0) # b,c,z,y,x
        output_k = []
        if verbose: print('patch {}: {}'.format(k, (z, y, x)))
        with torch.no_grad():
            for m in range(num_mirrors):
                if m == 1 and (2 in mirror_axes):
                    tta_dims = (4,)
                if m == 2 and (1 in mirror_axes):
                    tta_dims = (3,)
                if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                    # tta_dims = (4,3)
                    tta_dims = (3,4)
                if m == 4 and (0 in mirror_axes):
                    tta_dims = (2,)
                if m == 5 and (0 in mirror_axes) and (2 in mirror_axes): # should be (2,4)?
                    # tta_dims = (4,2)
                    tta_dims = (2,4)
                if m == 6 and (0 in mirror_axes) and (1 in mirror_axes): # should be (2,3)?
                    # tta_dims = (3,2)
                    tta_dims = (2,3)
                if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes): # should be (2,3,4)?
                    # tta_dims = (4,3,2)
                    tta_dims = (2,3,4)
                if m == 0:
                    tta_m = model_forward_pass(data=data_k, net=net, amp=amp)
                else:
                    tta_m = model_forward_pass(data=torch.flip(data_k, dims=tta_dims), net=net, amp=amp)
                    tta_m = torch.flip(tta_m, dims=tta_dims)
                tta_m = nonlin(tta_m)#.squeeze(dim=0)
                output_k.append(tta_m)
            # output_k = np.concatenate(output_k, axis=0).mean(axis=0)
            # output_k[:, :] *= importance_map
            output_k = torch.cat(output_k, dim=0).mean(dim=0)

            z = z - voi[0]
            y = y - voi[1]
            x = x - voi[2]
            output[:, z:z + pz, y:y + py, x:x + px] += output_k#.cpu().numpy()
            n_predictions[:, z:z + pz, y:y + py, x:x + px] += importance_map#_cpu

    output = output / n_predictions
    del n_predictions

    return output

if __name__ == '__main__':
    # create_case_list()
    pass