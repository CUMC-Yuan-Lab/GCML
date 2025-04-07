import argparse, logging
import os, time, glob
import numpy as np

from tqdm import tqdm
from importlib import import_module

import torch
import torch.nn.functional as F

from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from utils import setgpu, set_logger, init_weights, print_stdout_to_file, sliding_window_inference

import torch.nn as nn

from torch.utils.data import Dataset
import SimpleITK as sitk
import json

from batchgenerators.augmentations.spatial_transformations import augment_spatial
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_noise, augment_gaussian_blur
from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_gamma
from batchgenerators.augmentations.color_augmentations import augment_brightness_additive, augment_brightness_multiplicative
from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy

from scipy.stats import bernoulli

config_str = "config_vpanseg_demo"
lr_0=0.003
weight_decay_bool=0
batch_size_int=1
num_epochs=500

def img_crop_pad(img, crop, cval=None):

    sc, isz, isy, isx = img.shape
    z0, y0, x0, cz, cy, cx = crop
    x1 = x0 + cx
    y1 = y0 + cy
    z1 = z0 + cz
    crop_sz = crop[3:]

    res = np.ones([sc] + crop_sz, dtype=np.float32)

    mom = lambda x: max(0, x)
    non = lambda x, szx: x if x < szx else None

    # crop the valid region from the original image data
    temp = img[:, mom(z0):non(z1, isz), mom(y0):non(y1, isy), mom(x0):non(x1, isx)]

    assert(temp.shape[0] == sc)
    # print('img_crop_pad: ', crop)
    # expand if needed
    if temp.shape == res.shape:
        res = temp
    else:
        # print('expand...', crop, temp.min())
        # assert(0)
        if cval is None:
            res = temp.min() * res
        else:
            res = cval * res

        tz,ty,tx = temp.shape[1:]
        res[:, mom(-z0):mom(-z0)+tz, mom(-y0):mom(-y0)+ty, mom(-x0):mom(-x0)+tx] = temp

    return res

def train_img_augment3d(imb, c):
    assert isinstance(imb, tuple) and len(imb) == 2
    img0, label0 = imb

    # randomly sample the original  data to crop size
    img_sz = img0.shape[1:]
    crop_sz = c['patch_size']
    voi_sz = tuple(c['final_patch_size']) #tuple([c['crop_szz'], c['crop_szy'], c['crop_szx']])
    crop_margin = (np.array(crop_sz) - np.array(voi_sz)).astype(int)
    dim = len(img_sz)
    for d in range(dim):
        if crop_margin[d] + img_sz[d] < crop_sz[d]:
            crop_margin[d] = crop_sz[d] - img_sz[d]

    lbs = [-crop_margin[d] // 2 for d in range(dim)]
    ubs = [img_sz[d] + crop_margin[d] // 2 + crop_margin[d] % 2 - crop_sz[d] for d in range(dim)]


    cls = c['classes']  # including background = 0
    p_cls = c['class_sampling_prob']
    # css = np.random.choice(cls, size=c['batch_size'], p=p_cls)
    css = np.random.choice(cls, size=c['num_rois_per_case'], p=p_cls)
    # c_sample = np.random.choice(cls, size=None, p=p_cls)

    img_list = []
    lbl_list = []
    img_min = img0.min()
    # img_max = img0.max()

    for c_sample in css:
        # print('c_sample = ', c_sample)
        if c_sample == 0:  # background 0:  # random
            crop_lbs = [np.random.randint(low=lbs[d], high=ubs[d] + 1) for d in range(dim)]
        else:
            idx = np.where(label0 == c_sample)
            n_idx = len(idx[1])
            if n_idx == 0:  # randomly pick one from foreground
                lbl_idx = np.where(label0 > 0)
                n_lbl_idx = len(lbl_idx[0])

                if n_lbl_idx > 0:
                    i = np.random.randint(low=0, high=n_lbl_idx, size=None)
                    crop_lbs = [max(lbs[d], lbl_idx[d + 1][i] - crop_sz[d] // 2) for d in range(dim)]
                else:
                    crop_lbs = [np.random.randint(low=lbs[d], high=ubs[d] + 1) for d in range(dim)]
            else:
                i = np.random.randint(low=0, high=n_idx, size=None)
                crop_lbs = [max(lbs[d], idx[d + 1][i] - crop_sz[d] // 2) for d in range(dim)]

        crop_voi = crop_lbs + list(crop_sz)

        img = img_crop_pad(img0, crop_voi, cval=img_min)
        # assert(np.max(abs(img_nnunet-img)) == 0.0)

        # label_nnunet = img_crop_pad_nnunetv2(label0, crop=crop_voi,  cval=0)
        label = img_crop_pad(label0, crop_voi, cval=0)
        # assert (np.max(abs(label_nnunet - label)) == 0.0)


        # crop and normalize img, YY 09/10/23
        img = np.clip(img, a_min=c['lower_bound'], a_max=c['upper_bound']).astype(np.float32)
        if c['z_transform'] is not None:
            img = (img - c['z_transform'][0]) / c['z_transform'][1]
        else:
            img = img / max(abs(c['lower_bound']), abs(c['upper_bound']))

        rotate = c['rotate']
        if rotate is None:
            rotate = {'p':0.0, 'range':(0, 0)}
        scale = c['scale']
        if scale is None:
            scale = {'p':0.0, 'range':(1, 1)}

        img = np.expand_dims(img, axis=0)
        # label = np.swapaxes(np.expand_dims(label, axis=0), 1, 3)
        label = np.expand_dims(label, axis=0)

        # swap axis to make [x,y,z] order
        img = np.swapaxes(img, 2, 4)
        label = np.swapaxes(label, 2, 4)
        voi_sz = voi_sz[::-1]


        img, label = augment_spatial(img, label, voi_sz,
                                     patch_center_dist_from_border=30, # not used for random_crop=False
                                     do_elastic_deform=False,
                                     alpha=(0., 900.), sigma=(9., 13.),
                                     do_rotation=True,
                                     p_rot_per_sample=rotate['p'],
                                     angle_x=(
                                     rotate['range'][0] / 360. * 2. * np.pi, rotate['range'][1] / 360. * 2. * np.pi),
                                     angle_y=(
                                     rotate['range'][0] / 360. * 2. * np.pi, rotate['range'][1] / 360. * 2. * np.pi),
                                     angle_z=(
                                     rotate['range'][0] / 360. * 2. * np.pi, rotate['range'][1] / 360. * 2. * np.pi),
                                     do_scale=True,
                                     p_scale_per_sample=scale['p'],
                                     scale=scale['range'],
                                     border_mode_data='constant',
                                     border_cval_data=img.min(),  # this may cause discontinuity for both img and label
                                     order_data=3,
                                     border_mode_seg='constant',
                                     border_cval_seg=0,
                                     order_seg=1,
                                     random_crop=False
                                     )

        img = img.squeeze(axis=0)
        label = label.squeeze(axis=0)

        # assert(0)

        if c['gaussian_noise'] is not None: # default: gn['p'] = 0.1, gn['range'] = (0, 0.1)
            gn = c['gaussian_noise']
            if np.random.uniform() < gn['p']:
                img = augment_gaussian_noise(img, noise_variance=gn['range'], p_per_channel=1)

        if c['gaussian_blur'] is not None: # default: gb['p'] = 0.2, gb['range'] = (0.5, 1.0)
            gb = c['gaussian_blur']
            if np.random.uniform() < gb['p']:
                img = augment_gaussian_blur(img, sigma_range=gb['range'], different_sigma_per_axis=True, p_per_channel=0.5)

        if c['bright_mul'] is not None: # default: bm['p'] = 0.15, bm['range'] = (0.75, 1.25)
            bm = c['bright_mul']
            if np.random.uniform() < bm['p']:
                img = augment_brightness_multiplicative(img, multiplier_range=bm['range'], per_channel=True)

        # shift intensity range by 0.1 * std = 0.1
        shift = c['shift']
        if shift is not None:
            if np.random.uniform() < shift['p']:
                img = augment_brightness_additive(img, mu=shift['range'][0], sigma=shift['range'][1], per_channel=True)

        # assert(0)
        contrast = c['contrast']
        if contrast is not None:
            if np.random.uniform() < contrast['p']:
                img = augment_contrast(img, contrast_range=contrast['range'], preserve_range=True, per_channel=True)
        # print('after contrast img.shape = {},min ={}, max = {}'.format(img.shape, img.min(), img.max()))

        if c['low_resolution'] is not None:  # default: lowr['p'] = 0.25, lowr['range'] = (0.5, 1.0)
            lowr = c['low_resolution']
            if np.random.uniform() < lowr['p']:
                img = augment_linear_downsampling_scipy(img, zoom_range=lowr['range'], per_channel=True,
                                                        p_per_channel=0.5, order_downsample=0, order_upsample=3)

        gamma_invert = c['gamma_invert']
        if gamma_invert is not None:
            if np.random.uniform() < gamma_invert['p']:
                img = augment_gamma(img, gamma_range=gamma_invert['range'], invert_image=True, epsilon=1e-7,
                                    per_channel=True, retain_stats=True)
        gamma = c['gamma']
        if gamma is not None:
            if np.random.uniform() < gamma['p']:
                img = augment_gamma(img, gamma_range=gamma['range'], invert_image=False, epsilon=1e-7, per_channel=True,
                                    retain_stats=True)

        # flip sup-inf
        p_si = c['p_si']
        if p_si > 0:
            v = bernoulli.rvs(p_si, size=1)
            if v == 1:
                # print('p_si. img.shape = {}'.format(img.shape))
                # assert(0)
                img = np.flip(img, axis=-1).copy()
                if isinstance(imb, tuple):
                    label = np.flip(label, axis=-1).copy()

        # flip ant-post
        p_ud = c['p_ud']
        if p_ud > 0:
            v = bernoulli.rvs(p_ud, size=1)
            if v == 1:
                # print('flip ud...')
                img = np.flip(img, axis=-2).copy()
                if isinstance(imb, tuple):
                    label = np.flip(label, axis=-2).copy()

        # flip left-right
        p_lr = c['p_lr']
        if p_lr > 0:
            v = bernoulli.rvs(p_lr, size=1)
            if v == 1:
                # print('flip lr...')
                img = np.flip(img, axis=-3).copy()
                if isinstance(imb, tuple):
                    label = np.flip(label, axis=-3).copy()
        p_rot90 = c['p_rot90']
        if p_rot90 > 0:
            v = bernoulli.rvs(p_rot90, size=1)
            if v == 1:
                num_rot = np.random.choice((-3, -2, -1), size=1)
                axes = np.random.choice((-3, -2, -1), size=2, replace=False)

                img = np.rot90(img, k=num_rot, axes=axes).copy()
                if isinstance(imb, tuple):
                    label = np.rot90(label, k=num_rot, axes=axes).copy()

        # change img/label back from [nc,sx,sy,sz] to [nc, sz,sy,sx]
        img = np.swapaxes(img, 1, 3)
        label = np.swapaxes(label, 1, 3)
        voi_sz = voi_sz[::-1]

        img_list.append(img)
        lbl_list.append(label)

    img = np.asarray(img_list)#.squeeze(axis=0)
    label = np.asarray(lbl_list)#.squeeze(axis=0)

    # print(img.shape, label.shape)
    # assert(0)

    return img, label

def valid_img_augment3d(imb, c):

    img, label = imb
    # assert(0)
    # print('pre', img.min(), img.max())
    nch, iszz, iszy, iszx = img.shape
    cz = c['crop_szz']
    cy = c['crop_szy']
    cx = c['crop_szx']
    psz = c['final_patch_size']
    isz = img.shape[1:]

    # normalize img, YY 09/10/23
    img = np.clip(img, a_min=c['lower_bound'], a_max=c['upper_bound']).astype(np.float32)
    if c['z_transform'] is not None:
        img = (img - c['z_transform'][0]) / c['z_transform'][1]
    else:
        img = img / max(abs(c['lower_bound']), abs(c['upper_bound']))

    # if img_size < patch_size, do expansion, YY on 07/26/24
    img_sz = isz
    crop_size = [cz, cy, cx]
    crop_margin = np.asarray([0,0,0]).astype(int)
    dim = len(img_sz)
    need_expansion = False
    voi_size = list(img_sz)
    for d in range(dim):
        if img_sz[d] < crop_size[d]:
            crop_margin[d] = crop_size[d] - img_sz[d]
            voi_size[d] = crop_size[d]
            need_expansion = True
    lbs = [-crop_margin[d] // 2 for d in range(dim)]
    if need_expansion:
        expand_roi = lbs + list(voi_size)
        # print('before expansion: ', img.shape, label.shape)
        img = img_crop_pad(img, crop=expand_roi, cval=None)
        label = img_crop_pad(label, expand_roi, cval=0)
        # print('after expansion: ', img.shape, label.shape)

        isz = img.shape[1:]

    if not c['val_sliding_window']:
        # raise ValueError('please set val_sliding_window = True')

        # print('clss = ', clss)
        image_vois = []
        label_vois = []
        img_sz = isz
        crop_sz = [cz, cy, cx]
        dim = 3
        lbs = [0, 0, 0]
        ubs = [img_sz[d] - crop_sz[d] for d in range(dim)]

        cls = c['classes']  # including background = 0
        p_cls = c['class_sampling_prob']
        c_samples = np.random.choice(cls, size=c['n_test'], p=p_cls)
        for c_sample in c_samples:
            if c_sample == 0:  # background 0:  # random
                crop_lbs = [np.random.randint(low=lbs[d], high=ubs[d]+1) for d in range(dim)]
            else:
                idx = np.where(label == c_sample)
                n_idx = len(idx[1])
                if n_idx == 0:  # randomly pick one from foreground
                    lbl_idx = np.where(label > 0)
                    n_lbl_idx = len(lbl_idx[0])

                    if n_lbl_idx > 0:
                        i = np.random.randint(low=0, high=n_lbl_idx, size=None)
                        crop_lbs = [max(lbs[d], lbl_idx[d + 1][i] - crop_sz[d] // 2) for d in range(dim)]
                    else:
                        crop_lbs = [np.random.randint(low=lbs[d], high=ubs[d]) for d in range(dim)]
                else:
                    i = np.random.randint(low=0, high=n_idx, size=None)
                    crop_lbs = [max(lbs[d], idx[d + 1][i] - crop_sz[d] // 2) for d in range(dim)]

            crop_voi = crop_lbs + list(crop_sz)
            image_vois.append(img_crop_pad(img, crop=crop_voi, cval=None))
            label_vois.append(img_crop_pad(label, crop=crop_voi, cval=0))

        img = np.asarray(image_vois)
        label = np.asarray(label_vois)
    else:
        # select the VOI around all the foreground to save some time!
        if not c['full_image_inference']:
            lbl_idx = np.where(label > 0)
            n_lbl_idx = len(lbl_idx[0])
            if n_lbl_idx > 0:
                lbl_min = [lbl_idx[d+1].min() for d in range(3)]
                voi_lbs = [max(0, lbl_min[d] - psz[d]//4) for d in range(3)]
                voi_ups = [min(lbl_idx[d+1].max() + psz[d]//4, isz[d]) for d in range(3)]
                voi_sz = np.asarray(voi_ups) - np.asarray(voi_lbs)
                need_pad = [max(0, psz[d] - voi_sz[d]) for d in range(3)]
                if max(need_pad) > 0:
                    max_sz = [voi_sz[d] if voi_sz[d] > psz[d] else psz[d] for d in range(3)]
                else:
                    max_sz = list(voi_sz)
                crop_lbs = [voi_lbs[d] - need_pad[d] // 2 for d in range(3)]
                crop_voi = crop_lbs + max_sz
                # print('before crop: ', img.shape, label.shape, crop_voi)
                img = img_crop_pad(img, crop=crop_voi, cval=None)
                label = img_crop_pad(label, crop=crop_voi, cval=0)
                # print('after crop: ', img.shape, label.shape)

                # assert(0)
            else:
                x0 = int(iszx / 2 - cx / 2)
                y0 = int(iszy / 2 - cy / 2)
                z0 = int(iszz - cz - 1)

                cx = int(cx)
                cy = int(cy)
                cz = int(cz)

                img = img[:, z0:z0 + cz, y0:y0 + cy, x0:x0 + cx]
                label = label[:, z0:z0 + cz, y0:y0 + cy, x0:x0 + cx]

    # print('post: ', img.shape,label.shape, img.min(), img.max())
    # assert (0)
    return img, label

def get_patch_size(final_patch_size, rot_z, rot_y, rot_x, scale_range):
    '''
    final_patch_size = [z,y,x]
    '''

    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)
    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d

    coords = np.array(final_patch_size)[::-1]
    # print(coords)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
        # final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, rot_y, rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)[::-1]

def read_data(pt, s, c, phase='train', read_npy=False):
    data_dir = s['data_dir']
    npy_pt = pt.split(sep='.')[0] + '.npy'
    if read_npy:
        npy_data_file = os.path.join(data_dir + 'imagesTr/', npy_pt)
        data = np.load(npy_data_file, mmap_mode='r')
        #if data.dim == 3:
        #    data = np.expand_dims(data, axis=0)
        if phase == 'train':
            npy_lbl_file = os.path.join(data_dir, 'labelsTr/', npy_pt)
            label = np.load(npy_lbl_file,  mmap_mode='r')
            #if data.dim == 3:
            #    label = np.expand_dims(label, axis=0)
            return data, label
        else:
            return data
    else:
        img_file = os.path.join(data_dir + 'imagesTr/', pt)
        img = sitk.ReadImage(img_file)
        data = sitk.GetArrayFromImage(img)
        data = np.expand_dims(data, axis=0)

        if phase == 'train':
            lbl_file = os.path.join(data_dir + 'labelsTr/', pt)
            lbl = sitk.ReadImage(lbl_file)
            label = sitk.GetArrayFromImage(lbl)
            label = np.expand_dims(label, axis=0)
            # label = np.expand_dims(sitk.GetArrayFromImage(lbl), axis=0)
            return data, label
        else:
            return data

class DatasetSeg3Djson(Dataset):
    def __init__(self, s, c, phase='train', folds=None, n_classes=1):

        super(DatasetSeg3Djson, self).__init__()
        assert (phase == 'train' or phase == 'test')

        self.c = c
        self.s = s

        # load nnUNet split json file
        split_json_file = self.s['fname_list']
        with open(split_json_file, "r") as jf:
            jsonlist = json.loads(jf.read())

        n_cv = folds[0]
        f = folds[1]

        if f >= 0 and f <= n_cv - 1:
            te_ptnames = jsonlist[f]["val"]
            tr_ptnames = jsonlist[f]["train"]
        else:
            te_ptnames = []
            for i in range(n_cv):
                te_ptnames += jsonlist[i]["val"]
            tr_ptnames = te_ptnames

        # add suffix ".nii.gz" to each patient name
        tr_ptnames = [pt + ".nii.gz" for pt in tr_ptnames]
        te_ptnames = [pt + ".nii.gz" for pt in te_ptnames]
        # end of loading json file

        if phase == 'train':
            self.ptnames = tr_ptnames
            if c['augment_fcn'] is 'train_img_augment3d':
                self.transform = lambda x: train_img_augment3d(x, c=self.c) # 20231015
            else:
                raise ValueError(c['augment_fcn'] + ' is not implemented')
        else:
            self.ptnames = te_ptnames
            self.transform = lambda x: valid_img_augment3d(x, c=self.c)

        # self.transform = transform
        self.n_classes = n_classes
        self.c = c
        self.weight_sample = c['weight_sample']
        self.n_data = len(self.ptnames)

        self.phase = phase
        self.copy_label = c['copy_label']
        if c['preload_data']:
            pt_data = []
            for i in range(self.n_data):
                pt = self.ptnames[i]
                # print('preloading patient: #{}: {}'.format(i+1,pt))
                image, label = read_data(pt, s=self.s, c=self.c, phase='train', read_npy=True)
                pt_i = {
                    'name': pt,
                    'image': image,
                    'label': label
                }
                pt_data.append(pt_i)
            self.ptnames = pt_data


    def __getitem__(self, idx):
        pt = self.ptnames[idx]
        if self.c['preload_data']:
            image = pt['image']
            label = pt['label']
            pt = pt['name']
        else:
            #print('pt = {}'.format(pt))
            image, label = read_data(pt, s=self.s, c=self.c, phase='train', read_npy=True)

        if self.transform:
            sample = (image, label)
            image, label = self.transform(sample)

        ncls = len(self.c['OARs']) + 1
        if self.c['exclude_labels'] is not None:
            for l in self.c['exclude_labels']:
                label[np.where(label == l)] = 0
        assert(label.max() <= ncls)
        #if label.min() < 0:
        label = np.clip(label, a_min=0, a_max=None)
        assert(label.min() >= 0)

        if self.c['weight_sample'] is None:
            n_pixels = np.prod(label.shape[-3:]) # label.shape[0] * label.shape[1]
            weight = 1.0 * np.asarray([max(np.count_nonzero(label == i), 100) for i in range(ncls)], dtype=np.float32)
            # print('label size= {}'.format(weight))
            weight = np.sqrt(weight)
            weight[np.nonzero(weight)] = n_pixels / weight[np.nonzero(weight)]
        else:
            weight = np.asarray(self.c['weight_sample'],dtype=np.float32)

        weight = weight / weight.sum()
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        if self.phase == 'train':
            return (image, label, weight)
        else:
            return (image, label, pt)

    def __len__(self):
        return self.n_data

    def len(self):
        return self.n_data

def get_loss(loss_name, dp = False, dp_scale_factor=None, need_downsampling=True):
    '''
    return loss function
    :param loss_name:
    :param dp: if deep supervision is enabled
    :return:
    '''
    if loss_name is 'gja_new':
        criterion = GeneralizedJaccardNewLoss()
    elif loss_name is 'gja_ce_new':
        criterion = GeneralizedJaccardAndCrossEntropyNewLoss()
    else:
        err_msg = loss_name + ' is not implemented yet!'
        raise ValueError(err_msg)
    if dp is True:
        return DeepSupervision(criterion=criterion, dp_scale_factor=dp_scale_factor, need_downsampling=need_downsampling)
    else:
        return criterion

class DeepSupervision(nn.Module):
    def __init__(self, criterion, dp_scale_factor=None,  need_downsampling=True):
        '''
        copied from DeepSupervision on 8/19/23, set weight for each ds-level as 2**(-i) for i in range(4)
        target is down-sampled based on the level of ds (2**(-i))
        '''
        super(DeepSupervision, self).__init__()
        self.criterion = criterion
        self.scale_factors = dp_scale_factor
        self.need_downsampling = need_downsampling

    def forward(self, input, target, weight=None, nonlin=lambda x: x, background=True, ss=1):
        assert(isinstance(input, list))
        n_inputs = len(input)
        loss0 = lossdp = 0
        sf_sum = 0
        if n_inputs > 1:
            for i in range(n_inputs):
                if i == 0:
                    loss_i = self.criterion(input[i], target, weight=weight, nonlin=nonlin, background=background,
                                            ss=ss)
                    loss0 = loss_i[0]
                    sf_sum += 1
                else:
                    if self.scale_factors is None:
                        scale_factor = [2**(-i)] * 3
                    else:
                        scale_factor = self.scale_factors[i-1]
                    if  self.need_downsampling:
                        target_i = F.interpolate(target, scale_factor=scale_factor, mode='nearest-exact')#, align_corners=True)
                        loss_i = self.criterion(input[i], target_i, weight=weight, nonlin=nonlin, background=background,
                                            ss=ss)
                    else:
                        loss_i = self.criterion(input[i], target, weight=weight, nonlin=nonlin, background=background,
                                                ss=ss)
                    min_scale_factor = min(scale_factor)
                    lossdp += min_scale_factor * loss_i[0] # (n_inputs - i) * loss_i[0] # loss_i[0]
                    sf_sum += min_scale_factor

            loss = (loss0 + lossdp) / sf_sum # used for amos2023 # sf_sum = 1.875 (1 + 1/2 + 1/4 + 1/8 + 1/16)

            return [loss, loss0.item(), lossdp.item()]
        else:
            loss = self.criterion(input[0], target, weight, ss)
            return loss

class GeneralizedJaccardNewLoss(nn.Module):
    def __int__(self):
        super(GeneralizedJaccardNewLoss, self).__init__()
        # self.ss = ss

    def forward(self, input, target, weight = None, nonlin=lambda x: x, background=True, ss=1):
        '''
        with weight_i = 1/sum(c_i)
        :param input: (batch_size, n_classes, szy, szx)
        :param target: (batch_size, 1, szy, szx)
        :return:
        '''

        input = nonlin(input)

        bsz = input.shape[0] # batch_size
        ncls = input.shape[1]
        # n_pixels = target.size(2) * target.size(3) * target.size(4)
        target = target.type(torch.cuda.LongTensor)
        # one_hot = torch.cuda.FloatTensor(target.size(0), ncls, target.size(2), target.size(3), target.size(4)).zero_()
        one_hot = torch.cuda.FloatTensor(input.size()).zero_() # always cuda:0
        one_hot = one_hot.to(target.device) # fixed for GPU-swapping FL by JC on 24/08/20
        target = one_hot.scatter_(1, target.data, 1)
        target = target.view(bsz, ncls, -1)
        input = input.view(bsz, ncls, -1)
        if background == False:
            target = target[:,1:,:]
            input = input[:,1:,:]
            if weight is not None:
                # print(weight.shape)
                # print(weight)
                weight = weight[:,1:]
                weight = weight/weight.sum()
            ncls -= 1

        intersection = input * target
        # print(intersection.size(), input.size(), target.size(), ss)
        # assert(0)
        tp = intersection.sum(dim=(2,0))
        tsum = target.sum(dim=(2,0))
        psum = input.sum(dim=(2,0))
        jaccard = 1.0 - (tp + ss) / (tsum + psum - tp + ss)
        if weight is not None:
            loss_output = (jaccard * weight).sum()
        else:
            loss_output = jaccard.mean()
        return [loss_output]

class GeneralizedJaccardAndCrossEntropyNewLoss(nn.Module):
    def __int__(self):
        super(GeneralizedJaccardAndCrossEntropyNewLoss, self).__init__()

    def forward(self, input, target, weight = None, nonlin=lambda x: x, background=True, ss=10):
        '''
        with weight_i = 1/sum(c_i)
        :param input: (batch_size, n_classes, szy, szx)
        :param target: (batch_size, 1, szy, szx)
        :return:
        '''
        # print(input.shape, target.shape, type(input), type(target))
        # assert(0)
        criterion_gja = GeneralizedJaccardNewLoss()
        criterion_ce  = MyCrossEntropyNewLoss()

        # gja = criterion_gja(F.softmax(input, dim=1), target, weight, ss)
        gja = criterion_gja(input, target, weight, nonlin, background, ss)
        ce = criterion_ce(input, target)
        loss_output = 0.5 * gja[0] + 0.5 * ce[0]
        # assert(0)
        return [loss_output, gja[0].item(), ce[0].item()]

class MyCrossEntropyNewLoss(nn.Module):
    def __int__(self):
        super(MyCrossEntropyNewLoss, self).__init__()
        # self.ss = ss

    def forward(self, input, target, weight = None, ss=10, istrain=True):
        '''
        with weight_i = 1/sum(c_i)
        :param input: (batch_size, n_classes, szy, szx)
        :param target: (batch_size, 1, szy, szx)
        :return:
        '''
        # print(input.size(), target.size(), input.max(), input.min())
        # assert(0)
        target = torch.squeeze(target, dim=1)
        # print(target.size())

        target = target.type(torch.cuda.LongTensor)
        criterion = nn.CrossEntropyLoss()
        loss_output = criterion(input, target)
        # print(loss_output)
        # assert(0)
        return [loss_output]

class CEDiceMetrics(nn.Module):
    def __int__(self):
        super(CEDiceMetrics, self).__init__()
        # self.ss = ss

    def forward(self, input, target, background=False):
        '''
        with weight_i = 1/sum(c_i)
        :param input: (batch_size, n_classes, szz, szy, szx)
        :param target: (batch_size, 1, szz, szy, szx)
        :return:
        '''

        # one-hot
        # print target.dim()

        # input = F.softmax(input, dim=1)
        with torch.no_grad():
            bsz = input.shape[0] # batch_size
            ncls = input.shape[1]

            one_hot_target = torch.cuda.FloatTensor(input.size()).zero_()
            one_hot_target = one_hot_target.to(target.device)  # fixed for GPU-swapping FL by JC on 24/08/20
            target = target.type(torch.cuda.LongTensor)
            # print('target: ', torch.unique(target))
            # print('before target: ', target.size())
            target = one_hot_target.scatter_(1, target.data, 1)
            # print('after target: ', target.size())
            target = target.view(bsz, ncls, -1)

            one_hot_input = torch.cuda.FloatTensor(input.size()).zero_()
            one_hot_input = one_hot_input.to(input.device)  # fixed for GPU-swapping FL by JC on 24/08/20
            input = torch.argmax(input, dim=1, keepdim=True)
            # print(torch.unique(input))
            # print('before input: ', input.size())
            input = one_hot_input.scatter_(1, input.data, 1)
            input = input.view(bsz, ncls, -1)

            if background == False:
                target = target[:,1:,:]
                input = input[:,1:,:]

            intersection = input * target
            tp = intersection.sum(dim=2)
            psum = input.sum(dim=2)
            tsum = target.sum(dim=2)
            # dice = 2.0 * intersection.sum(dim=2) / (input.sum(dim=2) + target.sum(dim=2) + 1e-5) # [batch_size, n_classes]
            dice = 2.0 * tp / (psum + tsum + 1e-5)  # [batch_size, n_classes]
            # print(dice.size(), tp.size(), psum.size(), tsum.size())
            # assert(0)
        # dice = dice.mean(dim=0)

        return [dice.detach().cpu().numpy(),
                tp.detach().cpu().numpy(), psum.detach().cpu().numpy(), tsum.detach().cpu().numpy()]

def train(data_loader, net, loss, c, optimizer, lr, amp_grad_scaler=None):

    net.train()
    # lr = set_lr(epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    metrics = []

    # output_prob = []
    # clss = []
    weights = []
    if c['orl']:
        nonlin = lambda x: F.sigmoid(x)# c['nonlin_output'] #lambda x: F.softmax(x, dim=1)
    else:
        nonlin = lambda x: F.softmax(x, dim=1)
    # print(os.environ['CUDA_VISIBLE_DEVICES'])
    # assert(0)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    cz = c['crop_szz']
    cy = c['crop_szy']
    cx = c['crop_szx']
    for (data, target, weight) in tqdm(data_loader):
        if data.dim() == 6:
            data = data.view([-1,data.shape[-4], cz, cy, cx])
            target = target.view([-1, data.shape[-4], cz, cy, cx])
        weight = weight[0:1,:]

        if c['weight_sample'] is None:
            weights.append(weight.numpy())
        # print('cls = {}'.format(cls))
        # assert(0)
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        target = target.type(torch.cuda.FloatTensor)
        weight = weight.cuda(non_blocking=True)
        weight = weight.type(torch.cuda.FloatTensor)
        # assert (0)
        # automatic mixed presicion
        optimizer.zero_grad()
        if c['amp']:
            with autocast():
                output = net(data)
                if isinstance(output, list) and c['dp'] is False:
                    loss_output = loss(output[0], target, weight=weight,nonlin=nonlin,
                                       background=c['val_with_background'])
                else:
                    loss_output = loss(output, target, weight=weight, nonlin=nonlin,
                                       background=c['val_with_background'])

            amp_grad_scaler.scale(loss_output[0]).backward()
            if c['grad_norm_clip'] is not None :
                amp_grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=c['grad_norm_clip'])
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
        else:
            # print("data.shape = ", data.shape)
            output = net(data)
            if isinstance(output, list) and c['dp'] is False:
                loss_output = loss(output[0], target, weight=weight, nonlin=nonlin, background=c['val_with_background'])
            else:
                loss_output = loss(output, target, weight=weight, nonlin=nonlin, background=c['val_with_background'])

            loss_output[0].backward()
            if c['grad_norm_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=c['grad_norm_clip'])

            optimizer.step()

        loss_output[0] = loss_output[0].item()
        # print(weight, loss_output[0])
        metrics.append(loss_output)


    metrics = np.asarray(metrics, np.float32)
    # print metrics
    # assert(0)
    train_loss = np.mean(metrics, axis=0)

    return train_loss

def validate(data_loader, net, loss, c, amp_grad_scaler=None, save_output='None'):

    net.eval()
    metrics = []
    dice_list = []

    if c['orl']:
        nonlin = lambda x: F.sigmoid(x) # c['nonlin_output'] # lambda x: F.softmax(x, dim=1)
    else:
        nonlin = lambda x: F.softmax(x, dim=1)

    dice_metric = CEDiceMetrics()
    dice_metric = dice_metric.cuda()

    nch = len(c['img_ch']) # + np.asarray(c['one_hot_input'])


    with torch.no_grad():
        if c['val_sliding_window']:
            for i, (data, target, pt) in enumerate(data_loader):
                cz = np.asarray([c['crop_szz'], c['crop_szy'], c['crop_szx']])
                # cz = np.asarray([c['crop_szx'], c['crop_szy'], c['crop_szz']])
                ncls = len(c['OARs']) + 1
                voi = tuple(3 * [0]) + data.size()[-3:]
                # print('val_voi:', voi)
                # assert(0)
                patch_output = sliding_window_inference(data.squeeze(dim=0), voi, net, patch_size=cz, ncls=ncls,
                                                        step_ratio=0.5, importance_map=1.0,
                                                        nonlin=lambda x: x, ckpt_list=None, amp=c['amp'],
                                                        verbose=False).unsqueeze_(dim=0)

                target = target.cuda(non_blocking=True)

                dice = dice_metric(patch_output, target)[0]
                # print val dice to file
                if save_output is not 'None':
                    # dice = dice.item()
                    # print(pt, ' : ', dice)
                    outcsv_file = os.path.join(c['save_dir'], 'val_dice_{}_f{}.csv'.format(save_output, c['fold']))
                    outstr_i = '{},{:5.4f}\n'.format(pt[0], dice.item())
                    f = open(outcsv_file, 'a+')
                    f.write(outstr_i)
                    f.close()
                    # print('Saved val DICE of {} model to {}'.format(save_output, outcsv_file))

                dice_list.append(np.squeeze(dice))
                loss_output = 1.0 - dice.mean()
                # assert(0)
                # print(loss_output)

                del patch_output, target
                metrics.append(loss_output)

            metrics = np.asarray(metrics, np.float32)
            # print('metrics = ', metrics)
            # print('dice_list = ', dice_list)
            # assert(0)
            valid_loss = [np.mean(metrics, axis=0)]
            if len(dice_list) > 0:
                dice_list = np.asarray(dice_list, np.float32)
                valid_dice = np.mean(dice_list, axis=0, keepdims=True)
            else:
                valid_dice = None
            # print(valid_loss, valid_dice)
            # assert(0)

        else:
            # print('b')
            tps = []
            psums = []
            tsums = []
            for i, (data, target, weight) in enumerate(data_loader):
            # for (data, target, weight) in tqdm(data_loader):
                target = target.type(torch.FloatTensor)
                data = data.view(-1, nch, c['crop_szz'], c['crop_szy'], c['crop_szx'])
                target = target.view(-1, 1, c['crop_szz'], c['crop_szy'], c['crop_szx'])
                # print('after view:', data.shape, target.shape)
                # assert(0)
                n_vois = data.size()[0]

                for z in range(0, n_vois, 1):  # c['n_test']):
                    # print('z ', z)
                    z_end = min(z + 1, n_vois)
                    data_z = data[z:z_end]
                    target_z = target[z:z_end]
                    data_z = data_z.cuda()
                    target_z = target_z.cuda()
                    if data_z.dim() == 4:
                        data_z = torch.unsqueeze(data_z, dim=0)
                        target_z = torch.unsqueeze(target_z, dim=0)
                    # print(data.size(), target.size())
                    # assert(0)
                    if c['amp']:
                        with autocast():
                            output = net(data_z)
                    else:
                        output = net(data_z)

                    if isinstance(output, list) and c['dp'] is False:
                        output = output[0]
                    # print(output.size())
                    # assert(0)
                    dice_outputs = dice_metric(output, target_z)[1:]
                    del data_z, target_z
                    # print(dice_outputs[0].shape)
                    # assert(0)
                    tps.append(dice_outputs[0])
                    psums.append(dice_outputs[1])
                    tsums.append(dice_outputs[2])

            tps = np.concatenate(tps, axis=0)
            psums = np.concatenate(psums, axis=0)
            tsums = np.concatenate(tsums, axis=0)
            # print(tps.shape, psums.shape, tsums.shape)
            # assert (0)

            valid_dice = 2 * tps.sum(axis=0) / (psums.sum(axis=0) + tsums.sum(axis=0) + 1e-5)  # nclass x 1
            valid_loss = [1.0 - valid_dice.mean()]
    # torch.cuda.empty_cache()

    return valid_loss, valid_dice

def get_args():
    # global args
    parser = argparse.ArgumentParser(description='PyTorch 3D Deep Planning')
    parser.add_argument('-f', '--fold', type=int, metavar='F',
                        help='the fold for cross-validation when training models')
    parser.add_argument('--gpu', default='all', type=str, metavar='N',
                        help='the GPUs used (default: all)')
    args = parser.parse_args()
    return args

# set learn rate

def set_lr_step(epoch, prev_epoch, curr_lr, init_lr, num_epochs): # add num_epochs on 3/23/21 by YY
    # same as set_lr2 in seg3d_trainV2.py
    lr = curr_lr # = lr0
    if epoch < max(150, num_epochs/2):
        lr = init_lr # = lr_0
    else:
        patience_period = np.clip(num_epochs/10, a_min=30, a_max=50)
        if (epoch - prev_epoch) >= patience_period:
            lr = max(0.3 * curr_lr, 1e-6)
    return lr

def set_lr_cosine_warmup(epoch, init_lr, num_epochs):
    # num_epochs: 305
    lr_min = init_lr * 0.05 # init_lr = lr0 in seg3d_trainV2
    epoch0 = 5
    if epoch < epoch0:
        lr = lr_min + (epoch-1) * (init_lr )/(epoch0-1) # [lr_min, lr_min + lr0]
    else:
        epoch_t = epoch - epoch0
        total_t = num_epochs - epoch0
        lr = 0.5 * (1 + np.cos(epoch_t * np.pi / total_t)) * init_lr + lr_min
    return lr

def set_lr_cosine_annealing(epoch, init_lr, num_epochs, starting_epoch=0):

    lr_min = 1e-3 * init_lr # init_lr = lr0 in seg3d_trainV2
    t_i = max(30.0, 0.1 * num_epochs)
    if epoch < starting_epoch:
        lr = init_lr
    else:
        epoch_t = np.mod(epoch - starting_epoch, t_i)
        lr = lr_min + 0.5 * (init_lr - lr_min) * (1 + np.cos(epoch_t * np.pi / t_i))
    return lr

def set_lr_poly(epoch, max_epoch, init_lr, n_warmup=0, power=0.9):

    lr_min = init_lr * 0.05 # = lr0 in seg3d_trainV2

    if epoch < n_warmup:
        lr = lr_min + (1.0*epoch) * (init_lr - lr_min) / (n_warmup - 1)  # [lr_min, lr0]
    else:
        lr = max(init_lr * (1-(1.0*(epoch-n_warmup))/max_epoch)**power, 1.0 * init_lr/max_epoch)

    return lr

def update_config(config, args):
    '''
    update configuration and make back compatible with old configuration files
    Args:
        config: original configuration
        args: args from command input

    Returns:
        c: new configuration
    '''
    c = config

    tag = config_str[8:]
    c['tag'] = tag
    c['n_test'] = 1
    c['batch_size'] = batch_size_int

    if 'orl' not in c.keys():
        c['orl'] = ('orl' in c['loss'])
    if 'nonlin' not in c.keys():
        c['nonlin'] = 'relu'
    if 'dp' not in c.keys():
        c['dp'] = False
    if 'init' not in c.keys():
        c['init'] = 'xavier'
    if 'val_ma_alpha' not in c.keys():
        c['val_ma_alpha'] = 0.0
    if 'lr_decay' not in c.keys():
        c['lr_decay'] = 'step'
    if 'amp' not in c.keys():
        c['amp'] = False

    if 'prob_output' not in c.keys():
        c['prob_output'] = False
    if 'z_transform' not in c.keys():
        c['z_transform'] = None
    if 'val_sliding_window' not in c.keys():
        c['val_sliding_window'] = False
    if 'num_training_per_val' not in c.keys():
        c['num_training_per_val'] = 1
    if 'exclude_labels' not in c.keys():
        c['exclude_labels'] = None
    if 'copy_label' not in c.keys():
        c['copy_label'] = True
    if 'valid_start' not in c.keys():
        c['valid_start'] = 0.0

    if 'val_loss' not in c.keys():
        c['val_loss'] = c['loss']

    c['final_patch_size'] = [c['crop_szz'], c['crop_szy'], c['crop_szx']] # [z,y,x]
    c['patch_size'] = c['final_patch_size']
    if (c['rotate'] is not None) and (c['scale'] is not None):
        max_rot = np.max(np.abs(np.asarray(c['rotate']['range'])))
        rot_x = rot_y = rot_z = max_rot / 360.0 * 2. * np.pi
        scale_range = c['scale']['range']
        c['patch_size'] = get_patch_size(c['final_patch_size'], rot_z, rot_y, rot_x, scale_range=scale_range) # z,y,x

    c['classes'] = np.asarray([0] + list(c['OARs'].values()))
    n_classes = c['n_classes'] = len(c['classes'])

    if 'class_sampling_prob' not in c.keys():
        # default setting: background : foreground = 1 : 2 (borrowed from nnUNet)
        s_prob = np.asarray(n_classes * [1.0])
        s_prob[1:] = 2.0 * s_prob[1:] / (n_classes - 1)
        c['class_sampling_prob'] = s_prob / 3.0
    else:
        if c['class_sampling_prob'] is not None:
            c['class_sampling_prob'] = np.asarray(c['class_sampling_prob'])
        else:
            s_prob = np.asarray(n_classes * [1.0])
            s_prob[1:] = 2.0 * s_prob[1:] / (n_classes - 1)
            c['class_sampling_prob'] = s_prob / 3.0

    c['class_sampling_prob'] = c['class_sampling_prob'] / c['class_sampling_prob'].sum()

    if c['exclude_labels'] is None:
        assert(len(c['class_sampling_prob']) == n_classes)
    else:
        assert(len(c['class_sampling_prob']) == n_classes + len(c['exclude_labels']))

    # default image augmentation
    if 'gaussian_noise' not in c.keys():
        c['gaussian_noise'] = None
    if 'gaussian_blur' not in c.keys():
        c['gaussian_blur'] = None
    if 'bright_mul' not in c.keys():
        c['bright_mul'] = None
    if 'low_resolution' not in c.keys():
        c['low_resolution'] = None
    if 'gamma_invert' not in c.keys():
        c['gamma_invert'] = None

    if 'p_rot90' not in c.keys():
        c['p_rot90'] = 0.0
    if 'enable_dynamic_sampling_and_weights' not in c.keys():
        c['enable_dynamic_sampling_and_weights'] = False

    if isinstance(c['margin'], list):
        c['margin'] = np.asarray(c['margin'])

    c['compile_model'] = False

    c['batch_size'] = batch_size_int
    c['full_image_inference'] = False
    c['sample_fg'] = False

    # dp_scale_factors
    if 'dp_separate_z' not in c.keys():
        c['dp_separate_z'] = False

    if 'dp_scale_factors' not in c.keys():
        c['dp_scale_factors'] = []
        for i in range(c['depth']-2):
            if c['dp_separate_z']:
                scale_factor = [2**(-i), 2**(-i-1), 2**(-i-1)]
            else:
                scale_factor = [2**(-i-1)] * 3
            c['dp_scale_factors'].append(scale_factor)

    if 'num_rois_per_case' not in c.keys():
        c['num_rois_per_case'] = 1

    if 'preload_data' not in c.keys():
        c['preload_data'] = False

    if 'dp_need_downsampling' not in c.keys():
        c['dp_need_downsampling'] = True
    return c

def dynamic_sampling_and_weights(epoch, num_epochs, c, val_dice, stdout_file):

    if epoch == int(0.5 * num_epochs):
        c['class_sampling_prob'] = p = np.ones_like((c['classes'])) / c['n_classes']
        print_stdout_to_file('Epoch = {}, class_sample_prob = ['.format(epoch)
                             + ', '.join(['{:.4f}'.format(p[i]) for i in range(c['n_classes'])])
                             + ']', stdout_file)

    elif epoch == int(0.70 * num_epochs):
        # p = np.asarray([1.0] + list(val_dice))**(-3.0)
        p = np.asarray([1.0] + [max(0.5, val_dice[i]) for i in range(c['n_classes'] - 1)]) ** (-3.0)
        p = p / np.sum(p)
        c['class_sampling_prob'] = p
        print_stdout_to_file('Epoch = {}, class_sample_prob = ['.format(epoch)
                             + ', '.join(['{:.4f}'.format(p[i]) for i in range(c['n_classes'])])
                             + ']', stdout_file)

    elif epoch == int(0.85 * num_epochs) :
        c['weight_sample'] = p = c['class_sampling_prob'] # it should be the weight of each class WRONG NAME!
        print_stdout_to_file('Epoch = {}, weight_sample = ['.format(epoch)
                             + ', '.join(['{:.4f}'.format(p[i]) for i in range(c['n_classes'])])
                             + ']', stdout_file)
    return c

def main():

    args = get_args()
    n_gpu = setgpu(args.gpu)

    s = import_module('settings.SETTINGS_tr')
    s = s.setting

    config_training = import_module('configs.' + config_str)
    c = config_training.config

    f = args.fold  # cross validation on fold 0
    NCV = c['NCV']

    assert (f >= 0 and f <= NCV)


    model_name = c['model']
    # in_channels = 11
    dropout_p = c['dropout_p']
    # gn = c['group_norm']
    up_mode = c['up_mode']
    depth = c['depth']
    nf = c['nf']
    n_classes = len(c['OARs']) + 1

    c = update_config(config=c, args=args)

    tag = c['tag']

    nonlin = c['nonlin']
    if nonlin == 'relu':
        nonlin_w = 'relu'
    else:
        nonlin_w = 'leaky_relu'

    dp = c['dp']
    init = c['init']
    val_ma_alpha = c['val_ma_alpha']
    lr_decay = c['lr_decay']

    # automatic mixed precision (AMP)
    if c['amp']:
        amp_grad_scaler = GradScaler()
    else:
        amp_grad_scaler = None

    out_channels = n_classes
    if c['orl']:
        out_channels -= 1

    in_channels = len(c['img_ch'])

    # if in_channels == 2:
    #     in_channels = n_classes + 1 # one_hot format of initial mask as additonal input channels
    model = import_module('models.' + model_name)
    net = model.Net(in_channels=in_channels, nclasses=out_channels, nf=nf, relu=nonlin, up_mode=up_mode, dropout_p=dropout_p,
                        depth=depth, padding=True, group_norm=c['group_norm'], deep_supervision=c['dp'])

    if c['compile_model']:
        net = torch.compile(net)
    init_weights(net, init_type=init, nonlin=nonlin_w)

    save_dir = os.path.join(s['param_dir'], config_str)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # log file
    logger = logging.getLogger(c['prefix'])
    log_file = os.path.join(save_dir, c['prefix'] + '_tr_v{}_f{}.txt'.format(c['tag'], f))

    logger = set_logger(logger, log_file, log=False)

    # stdout file
    stdout_file = os.path.join(save_dir, 'stdout_f{}.txt'.format(f))

    start_epoch = 0
    best_loss_epoch = num_epochs
    best_val_loss = 9999.0
    best_loss_MA_epoch = num_epochs  # moving average
    val_loss_MA = None
    best_val_loss_MA = 100.0

    loss = get_loss(c['loss'], dp=dp, dp_scale_factor=c['dp_scale_factors'], need_downsampling=c['dp_need_downsampling'])
    # n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    net = net.cuda()
    loss = loss.cuda()

    if c['val_sliding_window']:
        val_loss = get_loss(c['val_loss'], dp=False)
    else:
        val_loss = get_loss(c['val_loss'], dp=dp)
    val_loss = val_loss.cuda()

    # vloss_str = c['val_loss']
    if 'val_with_background' not in c.keys():
        c['val_with_background'] = True

    net = DataParallel(net)

    if c['optim'] is 'adam':
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr = lr_0,
            weight_decay = weight_decay_bool,
            amsgrad=True
        )
    else:
        logger.info(c['optim'] + ' is not implemented yet')
        raise ValueError

    optim_str = c['optim']

    history = []
    logger.info('starting training fold ({})...'.format(f))
    hist_file = os.path.join(save_dir, 'history_f{}.ckpt'.format(f))

    if os.path.isfile(hist_file):
        print('training reset')
        for rf in glob.glob(os.path.join(save_dir, '*_f{}.*'.format(f))):
            os.remove(rf)
            print(' - removed ' + rf)

    lr = lr_0
    lr_from_history = False

    load_start = time.time()
    # fname = s['data_dir']


    n_j = 8 * n_gpu

    # load train data
    tr_dataset = DatasetSeg3Djson(s=s, c=c, phase='train', folds=(NCV, f), n_classes=n_classes)


    # tr_dataset = myH5PYDatasetSeg3DV2(s=s, c=c, phase='train', folds=(NCV, f), n_classes=n_classes)
    n_train = tr_dataset.len()
    batch_size = batch_size_int
    train_loader = DataLoader(tr_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=n_j,
                              pin_memory=True)

    if f == NCV:
        data_phase = 'train'
    else:
        data_phase = 'test'

    #load valid data

    te_dataset = DatasetSeg3Djson(s=s, c=c, phase=data_phase, folds=(NCV, f), n_classes=n_classes)

    # te_dataset = myH5PYDatasetSeg3DV2(s=s, c=c, phase=data_phase, folds=(NCV, f), n_classes=n_classes)
    n_test = te_dataset.len()
    test_loader = DataLoader(te_dataset,
                              batch_size = batch_size_int,
                              shuffle = False,
                              num_workers = n_j,
                              pin_memory=True)

    load_end = time.time()
    logger.info('took {:.3f} seconds to load data: n_train = {}, n_test = {}'.format(load_end-load_start, n_train, n_test))

    prev_epoch = best_loss_epoch

    log_start = min(num_epochs * 0.2, 60)
    # lr0 = lr

    vidx = 0

    print_stdout_to_file('\n---------- Start model training! ({}) ----------\n'.
                         format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), stdout_file)


    for epoch in range(start_epoch, num_epochs):

        run_valid = False
        if (epoch > c['valid_start'] * num_epochs - 5):
            run_valid = True

        if lr_decay is 'step':
            lr0 = lr
            if lr_from_history == False:
                if run_valid:
                    lr = set_lr_step(epoch, prev_epoch, curr_lr=lr0, init_lr=lr_0, num_epochs=num_epochs)
            else:
                lr_from_history = False
                prev_epoch = epoch
            if lr != lr0:
                prev_epoch = epoch
        elif lr_decay is 'poly':
            lr = set_lr_poly(epoch, num_epochs, init_lr=lr_0, n_warmup=0)
        elif lr_decay is 'cosine':
            lr = set_lr_cosine_warmup(epoch + 1, init_lr=lr_0, num_epochs=num_epochs)
        elif lr_decay is 'cosine_annealing':
            lr = set_lr_cosine_annealing(epoch, init_lr=lr_0, num_epochs=num_epochs)
        else:
            raise ValueError(lr_decay + ' is not implemented yet!')

        best_loss_changed = False
        best_loss_MA_changed = False

        print_stdout_to_file('', stdout_file)
        print_stdout_to_file('Epoch [{}/{}|{}] [v{}/f{}/g{}/vi{}/{}] (lr {:.6f}):'.
                             format(epoch, num_epochs - 1, prev_epoch, tag, f, args.gpu, vidx, optim_str, lr),
                             stdout_file)
        c['sample_fg'] = False
        for n_tr in range(c['num_training_per_val']):
            if c['num_training_per_val'] > 1 and n_tr == c['num_training_per_val'] - 1:
                c['sample_fg'] = True
            start_time = time.time()

            train_metrics = train(train_loader, net, loss, c, optimizer, lr=lr, amp_grad_scaler=amp_grad_scaler)
            # print_stdout_to_file(prt_str, stdout_file)
            train_loss = train_metrics[0]
            train_time = time.time() - start_time
            print_stdout_to_file(' - training loss ({:.2f} s):\t{:.4f}\t'.format(train_time, train_loss)
                  + '\t'.join(['{:.4f}'.format(train_metrics[i]) for i in range(1, len(train_metrics))]),
                                 stdout_file)

        # print(' - training loss ({:.2f} s):\t\t{:.6f}'.format(train_time, train_loss))

        start_time = time.time()
        valid_dice = None
        if run_valid:
            valid_metrics, valid_dice = validate(test_loader, net, val_loss, c, amp_grad_scaler=amp_grad_scaler)
            valid_loss = valid_metrics[0]
        else:
            valid_loss = history[-1][1]
            valid_metrics = history[-1][-2]
            valid_dice = history[-1][-1]

        valid_time = time.time() - start_time
        if val_loss_MA is None:
            val_loss_MA = valid_loss
        else:
            val_loss_MA = val_ma_alpha * val_loss_MA + (1.0 - val_ma_alpha) * valid_loss

        print_stdout_to_file(' - validing loss ({:.2f} s):\t{:.4f}\t{:.4f}|\t'.format(valid_time, val_loss_MA, valid_loss)
             + '\t'.join(['{:.4f}'.format(valid_metrics[i]) for i in range(0, len(valid_metrics))]),
                             stdout_file)

        if valid_dice is not None:
            print_stdout_to_file(
                ' - validing DICE ({:.2f} s):\t{:.4f}| '.format(valid_time, np.mean(valid_dice))
                + '\t'.join(['{:.4f}'.format(valid_dice[i]) for i in range(0, len(valid_dice))]),
                stdout_file)

        history.append((train_loss, valid_loss, train_metrics, valid_metrics, valid_dice))

        if (val_loss_MA <= best_val_loss_MA) and run_valid:
            best_loss_MA_changed = True
            best_loss_MA_epoch = epoch
            #best_val_loss = valid_loss
            best_val_loss_MA = val_loss_MA
            # prev_epoch = epoch
        if (valid_loss < best_val_loss) and run_valid:
            best_loss_changed = True
            best_loss_epoch = epoch
            best_val_loss = valid_loss
            # prev_epoch = epoch

        # else:
        print_stdout_to_file(' - BEST(val_loss_MA)[{:d}]:\t{:.4f}'.format(best_loss_MA_epoch, best_val_loss_MA),
                             stdout_file)
        print_stdout_to_file(' - BEST(val_loss)[{:d}]:\t\t\t\t{:.4f}'.format(best_loss_epoch, best_val_loss),
                             stdout_file)

        if best_loss_MA_changed or best_loss_changed:
            prev_epoch = epoch

        if c['enable_dynamic_sampling_and_weights']:
            c = dynamic_sampling_and_weights(epoch, num_epochs=num_epochs, c=c, val_dice=history[best_loss_epoch][4],
                                             stdout_file=stdout_file)

        # save ckpt
        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        optimizer_state_dict = optimizer.state_dict()

        if best_loss_changed:
            # print(os.path.join(save_dir, 'best_val_loss_f{}.ckpt'.format(f)))
            torch.save({
                'save_dir': save_dir,
                'save_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                'state_dict': state_dict,
                'args': args,
                'config': c,
                'setting': s},
                os.path.join(save_dir, 'best_val_loss_f{}.ckpt'.format(f))
            )
            if (epoch + 1) > log_start:
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
            # 'optimizer_state_dict': optimizer_state_dict,
            'args': args,
            'config': c,
            'setting': s,
            'lr': lr,
        }
        if amp_grad_scaler is not None:
            history_dict['amp_grad_scaler'] = amp_grad_scaler.state_dict()

        torch.save(history_dict, os.path.join(save_dir, 'history_f{}.ckpt'.format(f)))

        if (epoch - prev_epoch) > 60 and lr == 1e-6:
            break

if __name__ == '__main__':

    main()