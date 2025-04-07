# import sys
import os, logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('agg')
import random

import SimpleITK as sitk

from skimage import exposure
from scipy.stats import bernoulli

from batchgenerators.augmentations.spatial_transformations import augment_spatial
from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_gamma, augment_brightness_additive

def img_crop_pad(img, crop, cval=None):
    '''
    crop first, then pad if needed to save memory and improve speed
    :param img:
    :param crop: list [x0,y0,z0,sx,sy,sz]
    :param cval:
    :return:
    '''
    sc, isx, isy, isz = img.shape
    x0, y0, z0, cx, cy, cz = crop
    x1 = x0 + cx
    y1 = y0 + cy
    z1 = z0 + cz
    res = np.ones([sc] + crop[3:], dtype=np.float32)

    mom = lambda x: max(0, x)
    non = lambda x, szx: x if x < szx else None
    # crop
    temp = img[:, mom(x0):non(x1, isx), mom(y0):non(y1, isy), mom(z0):non(z1, isz)]
    assert(temp.shape[0] == sc)

    # expand if needed
    if temp.shape == res.shape:
        res = temp
    else:
        if cval is None:
            res = temp.min() * res
        else:
            res = cval * res

        tx,ty,tz = temp.shape[1:]
        res[:,mom(-x0):mom(-x0)+tx,mom(-y0):mom(-y0)+ty,mom(-z0):mom(-z0)+tz] = temp
    return res

def save_nrrd(img, img_name):
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


def voi_histogram_equalization(image, mask, number_bins=256):
    '''
    perform histogram-equalization for voxels within the given mask
    :param image:
    :param mask:
    :param number_bins:
    :return:
    '''
    idx = np.nonzero(mask)
    img_data = image[idx]

    hist, bins = np.histogram(img_data, number_bins, density=True)
    cdf = hist.cumsum()

    cdf = (number_bins-1) * cdf / cdf[-1]  # normalized
    hq_image = np.interp(image.flatten(), bins[:-1], cdf)
    hq_image = hq_image.reshape(image.shape)
    hq_image[np.where(mask==0)] = 0
    return hq_image #hq_image.reshape(image.shape)#, cdf


def overlap_similarities(pred, reference, label=1):

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    pred = pred == label
    overlap_measures_filter.Execute(reference, pred)

    similarites = {}
    similarites['dice'] = overlap_measures_filter.GetDiceCoefficient()
    similarites['jaccard'] = overlap_measures_filter.GetJaccardCoefficient()
    similarites['voe'] = 1.0 - similarites['jaccard']
    similarites['rvd'] = 1.0 - overlap_measures_filter.GetVolumeSimilarity()

    return similarites

def img_augument3d(imb, crop, shift=None, rotate=0, scale=0, normalize_pctwise=None, p_si=0, p_lr=0, p_ud=0, p_rot90=0,
                   contrast=None, gamma=None, valid_list=False, istest=False, mrgn=8):
    """
    default value for normalize_pctwise=(20,95)
    """
    if isinstance(imb, tuple) and len(imb) == 2:
        img, label = imb
    else:
        img = imb
        label = None

    nch, iszx, iszy, iszz = img.shape
    brain_mask = np.zeros((iszx, iszy, iszz), dtype=np.uint8)
    ch_stats = np.zeros((nch, 2),dtype=np.float32)
    orig_label_values = np.unique(label)
    for i in range(nch):
        brain_mask[np.where(img[i]>0)] = 1
        img[i] = (img[i] - np.mean(img[i])) / np.std(img[i])
    if normalize_pctwise is not None:
        raise ValueError('normalized_pitwise needs to be rewritten!')
        pclow, pchigh = normalize_pctwise
        if not istest:
            pclow = np.random.randint(pclow[0], pclow[1])
            pchigh = np.random.randint(pchigh[0], pchigh[1])
        vl, vh = np.percentile(img, (pclow, pchigh))
        img = exposure.rescale_intensity(img, in_range=(vl, vh))  # range: 0 - 1

    if shift is not None:
        img = augment_brightness_additive(img, mu=shift[0], sigma=shift[1], per_channel=True)

    if contrast is not None:
        img = augment_contrast(img, contrast_range=contrast, preserve_range=True, per_channel=True)

    if gamma is not None:
        img = augment_gamma(img, gamma_range=gamma, invert_image=False, epsilon=1e-7, per_channel=True,
                            retain_stats=False)

    iz, iy, ix, cz, cy, cx = crop
    if not istest:
        label = label + np.expand_dims(brain_mask, axis=0)
        if rotate > 0 and scale > 0:
            raise ValueError('Both rotate and scale should be set as 0 for BraTS challenge')

            patch_size = (cx, cy, cz)  # crop[3:] (cx,cy,cz)
            ch_min = np.min(img, axis=(1,2,3), keepdims=True)
            ch_max = np.max(img, axis=(1,2,3), keepdims=True)
            img = (img - ch_min) / (ch_max - ch_min)

            img = np.expand_dims(img, axis=0)
            # label = np.swapaxes(np.expand_dims(label, axis=0), 1, 3)
            label = np.expand_dims(label, axis=0)

            print('before spatial augment: img', img.shape, img.min(), img.max())
            print('before spatial augment: label', label.shape, np.unique(label))
            print('patch_size', patch_size)

            img, label = augment_spatial(img, label, patch_size,
                                         patch_center_dist_from_border=(cx / 2, cy / 2, cz / 2),
                                         do_elastic_deform=True,
                                         alpha=(0., 900.), sigma=(9., 13.),
                                         do_rotation=True,
                                         angle_x=(-rotate / 360. * 2. * np.pi, rotate / 360. * 2. * np.pi),
                                         angle_y=(-rotate / 360. * 2. * np.pi, rotate / 360. * 2. * np.pi),
                                         angle_z=(-rotate / 360. * 2. * np.pi, rotate / 360. * 2. * np.pi),
                                         do_scale=True,
                                         scale=(1.0 - scale, 1.0 + scale),
                                         random_crop=True
                                         )
            # img = img.clip(min=0, max=1.0)
            img = img.squeeze(axis=0)
            label = label.squeeze()
            # brain_mask[label>0] = 1
            # label = label - 1
            print('after spatial augment: img', img.shape, img.min(), img.max())
            print('    non_neg = {}, non_ge_1 = {}'.format(np.count_nonzero(np.where(img<0.0)),
                                                           np.count_nonzero(np.where(img>1.0))))
            print('after spatial augment: label', label.shape, np.unique(label))

        cls = np.unique(label)[1:]
        p_cls = 1.0 * cls / cls.sum()

        c_sample = np.random.choice(cls, size=1, p=p_cls)[0]
        if c_sample == 1: #brain
            idx = np.where(label > 0)

            brsx = max(idx[1].max() - idx[1].min() + 1, cx) + 2 * mrgn
            start_x = max(0, idx[1].min()-mrgn)
            end_x = min(iszx, start_x + brsx) - cx
            # assert(end_x >= start_x)
            if end_x < start_x + 2 * mrgn:
                start_x = max(0, end_x - 2 * mrgn)
            x = np.random.randint(low=start_x, high=end_x, size=1)[0]

            brsy = max(idx[2].max() - idx[2].min() + 1, cy) + 2 * mrgn
            start_y = max(0, idx[2].min() - mrgn)
            end_y = min(iszy, start_y + brsy) - cy
            if end_y < start_y + 2 * mrgn:
                start_y = max(0, end_y - 2 * mrgn)
            y = np.random.randint(low=start_y, high=end_y, size=1)[0]

            brsz = max(idx[3].max() - idx[3].min() + 1, cz) + 2 * mrgn
            start_z = max(0, idx[3].min() - mrgn)
            end_z = min(iszz, start_z + brsz) - cz
            if end_z < start_z + 2 * mrgn:
                start_z = max(0, end_z - 2 * mrgn)
            z = np.random.randint(low=start_z, high=end_z, size=1)[0]

        else:
            idx = np.where(label == c_sample)
            n_idx = len(idx[1])
            if n_idx == 0:
                print('c_sample = {}, label vlaues = {}, orig_label_values = {}'.format(c_sample, np.unique(label), orig_label_values))
                assert(0)
            i = np.random.randint(low=0, high=n_idx, size=1)[0]
            ctrx = idx[1][i]
            ctry = idx[2][i]
            ctrz = idx[3][i]
            assert(label[:,ctrx,ctry,ctrz] == c_sample)

            x = max(0, ctrx - cx / 2)
            if x + cx > iszx: x = iszx - cx
            y = max(0, ctry - cy / 2)
            if y + cy > iszy: y = iszy - cy
            z = max(0, ctrz - cz / 2)
            if z + cz > iszz: z = iszz - cz

        x = int(x)
        y = int(y)
        z = int(z)

        cx = int(cx)
        cy = int(cy)
        cz = int(cz)
        img = img[:, x:x + cx, y:y + cy, z:z + cz]
        label = label[:, x:x + cx, y:y + cy, z:z + cz]
        label[np.nonzero(label)] = label[np.nonzero(label)] - 1

    else:

        label_max = label.max()
        #print('label_max', label_max)
        idx = np.where(label == label_max)
        ctrx = (idx[1].max() + idx[1].min()) / 2
        ctry = (idx[2].max() + idx[2].min()) / 2
        ctrz = (idx[3].max() + idx[3].min()) / 2


        z0 = max(0, ctrz - cz / 2)
        if z0 + cz > iszz: z0 = iszz - cz
        y0 = max(0, ctry - cy / 2)
        if y0 + cy > iszy: y0 = iszy - cy
        x0 = max(0, ctrx - cx / 2)
        if x0 + cx > iszx: x0 = iszx - cx

        x0 = int(x0)
        y0 = int(y0)
        z0 = int(z0)

        cx = int(cx)
        cy = int(cy)
        cz = int(cz)
        img = img[:, x0:x0 + cx, y0:y0 + cy, z0:z0 + cz]
        label = label[:, x0:x0 + cx, y0:y0 + cy, z0:z0 + cz]

    # flip sup-inf
    if p_si > 0 and not istest:
        v = bernoulli.rvs(p_si, size=1)
        if v == 1:
            img = np.flip(img, axis=3).copy()
            if isinstance(imb, tuple):
                label = np.flip(label, axis=3).copy()

    # flip up-down
    if p_ud > 0 and not istest:
        v = bernoulli.rvs(p_ud, size=1)
        if v == 1:
            img = np.flip(img, axis=2).copy()
            if isinstance(imb, tuple):
                label = np.flip(label, axis=2).copy()

    # flip left-right
    if p_lr > 0 and not istest:
        v = bernoulli.rvs(p_lr, size=1)
        if v == 1:
            img = np.flip(img, axis=1).copy()
            if isinstance(imb, tuple):
                label = np.flip(label, axis=1).copy()

    if p_rot90 > 0 and not istest:
        v = bernoulli.rvs(p_rot90, size=1)
        if v == 1:
            num_rot = np.random.choice((1,2,3), size=1)
            axes = np.random.choice((1,2,3), size=2, replace=False)
            img = np.rot90(img, k=num_rot, axes=axes).copy()
            if isinstance(imb, tuple):
                label = np.rot90(label, k=num_rot, axes=axes).copy()


    if isinstance(imb, tuple):
        return img, label
    else:
        return img


# copied from hecktor20/
def img_augument3d_v2(imb, crop, shift=0, rotate=0, scale=0, normalize_pctwise=None, p_si=0, p_lr=0, p_ud=0,
                   contrast=None, gamma=None, valid_list=False, istest=False, mrgn=8):
    """
    default value for normalize_pctwise=(20,95)
    compared to img_augument3d, change the way of randomly cropping VOI when label of brain is selected
    YY, 06/13/20
    """
    if isinstance(imb, tuple) and len(imb) == 2:
        img, label = imb
    else:
        img = imb
        label = None
    # default input img: (z,y,x,c), range:[0,1]
    # obtain mean and std for each channel, and the brain mask
    # print('raw img: type = {}, shape = {}, min ={}, max = {}'.format(img.dtype, img.shape, img.min(), img.max()))
    # print('raw label:type = {},shape = {}, range = {}'.format(label.dtype, label.shape, np.unique(label)))
    # for i in np.unique(label)[1:]:
    #     print('label = {}, number of voxels = {}'.format(i, len(np.where(label==i)[0])))
    # assert(0)
    nch, iszx, iszy, iszz = img.shape
    brain_mask = np.zeros((iszx, iszy, iszz), dtype=np.uint8)
    ch_stats = np.zeros((nch, 2), dtype=np.float32)
    orig_label_values = np.unique(label)
    for i in range(nch):
        brain_mask[np.where(img[i] > 0)] = 1
        # ch_stats[i] = np.asarray([np.mean(img[i]), np.std(img[i])])
        # print('before scale: img: min = {}, max = {}, mean = {}, std = {}'.format(img[i].min(), img[i].max(), np.mean(img[i]), np.std(img[i])))
        img[i] = (img[i] - np.mean(img[i])) / np.std(img[i])
        # print('       scaled:img: min = {}, max = {}'.format(img[i].min(), img[i].max()))

    # print('ch_stats', ch_stats)
    # print('brain values = {}, size = {}'.format(np.unique(brain_mask), np.count_nonzero(brain_mask)))
    if normalize_pctwise is not None:
        raise ValueError('normalized_pitwise needs to be rewritten!')
        pclow, pchigh = normalize_pctwise
        if not istest:
            pclow = np.random.randint(pclow[0], pclow[1])
            pchigh = np.random.randint(pchigh[0], pchigh[1])
        vl, vh = np.percentile(img, (pclow, pchigh))
        img = exposure.rescale_intensity(img, in_range=(vl, vh))  # range: 0 - 1

    # print('raw img.shape = {},min ={}, max = {}'.format(img.shape, img.min(), img.max()))
    # print('raw label range = {}'.format(np.unique(label)))
    if contrast is not None:
        img = augment_contrast(img, contrast_range=contrast, preserve_range=True, per_channel=True)
    # print('after contrast img.shape = {},min ={}, max = {}'.format(img.shape, img.min(), img.max()))
    if gamma is not None:
        img = augment_gamma(img, gamma_range=gamma, invert_image=False, epsilon=1e-7, per_channel=False,
                            retain_stats=False)
    # print('after gamma img.shape = {},min ={}, max = {}'.format(img.shape, img.min(), img.max()))
    # assert(0)
    iz, iy, ix, cz, cy, cx = crop
    if not istest:
        label = label + np.expand_dims(brain_mask, axis=0)
        # print('after add brain mask, label.shape = {}, classes = {}'.format(label.shape, np.unique(label)))
        # for i in np.unique(label)[1:]:
        #     print('        label = {}, number of voxels = {}'.format(i, len(np.where(label == i)[0])))
        # assert(0)
        if rotate > 0 and scale > 0:
            # print rotate, scale
            patch_size = (cx, cy, cz)  # crop[3:] (cx,cy,cz)
            # print('patch_size = {}'.format(patch_size))
            '''
            basic_patch_size = get_patch_size(patch_size,
                                              rot_x = (-15./360 * 2. * np.pi, 15./360 * 2. * np.pi),
                                              rot_y = (-15./360 * 2. * np.pi, 15./360 * 2. * np.pi),
                                              rot_z = (-15./360 * 2. * np.pi, 15./360 * 2. * np.pi),
                                              scale_range = (0.85, 1.15))
            '''
            # print('basic_patch_size', basic_patch_size)
            # assert(0)
            # img = np.swapaxes(img, 1, 3)

            img = np.expand_dims(img, axis=0)
            # label = np.swapaxes(np.expand_dims(label, axis=0), 1, 3)
            label = np.expand_dims(label, axis=0)

            print('before spatial augment: img', img.shape, img.min(), img.max())
            print('before spatial augment: label', label.shape, label.min(), label.max())
            print('patch_size', patch_size)
            assert (0)
            img, label = augment_spatial(img, label, patch_size,
                                         patch_center_dist_from_border=(cx / 2, cy / 2, cz / 2),
                                         do_elastic_deform=True,
                                         alpha=(0., 900.), sigma=(9., 13.),
                                         do_rotation=True,
                                         angle_x=(-rotate / 360. * 2. * np.pi, rotate / 360. * 2. * np.pi),
                                         angle_y=(-rotate / 360. * 2. * np.pi, rotate / 360. * 2. * np.pi),
                                         angle_z=(-rotate / 360. * 2. * np.pi, rotate / 360. * 2. * np.pi),
                                         do_scale=True,
                                         scale=(1.0 - scale, 1.0 + scale),
                                         random_crop=False
                                         )
            # print(img.min(), img.max(), img.shape)
            # img = img.clip(min=0, max=1.0)
            img = img.squeeze(axis=0)
            label = label.squeeze()
            # brain_mask[label>0] = 1
            # label = label - 1
            print('after spatial augment: img', img.shape, img.min(), img.max())
            print('after spatial augment: label', label.shape, label.min(), label.max())

            # assert (0)
        # label should have labels as [1,2,3,4] in which 1 is the brain
        # cls = np.asarray([1,2,3,4],dtype=np.uint8)
        cls = np.unique(label)[1:]
        p_cls = 1.0 * cls / cls.sum()
        # print('cls = {}, p_cls = {}'.format(cls, p_cls))
        c_sample = np.random.choice(cls, size=1, p=p_cls)[0]
        # c_sample = 1
        # idx = np.where(label == c_sample)
        # print('c_sample:',c_sample)
        # if cls[-1] < 4:
        #     print(cls, p_cls, c_sample)
        # assert(0)
        # assert(0)

        idx = np.where(label == c_sample)
        n_idx = len(idx[1])
        # print('n_idx = {}'.format(n_idx))
        # assert(n_idx > 0)
        if n_idx == 0:
            print('c_sample = {}, label vlaues = {}, orig_label_values = {}'.format(c_sample, np.unique(label),
                                                                                    orig_label_values))
            assert (0)
        i = np.random.randint(low=0, high=n_idx, size=1)[0]
        ctrx = idx[1][i]
        ctry = idx[2][i]
        ctrz = idx[3][i]
        # print('the {} th voxel[{},{},{}] with label [{}] is selected'.format(i, ctrx, ctry, ctrz, c_sample))
        assert (label[:, ctrx, ctry, ctrz] == c_sample)

        x = max(0, ctrx - cx / 2)
        if x + cx > iszx: x = iszx - cx
        y = max(0, ctry - cy / 2)
        if y + cy > iszy: y = iszy - cy
        z = max(0, ctrz - cz / 2)
        if z + cz > iszz: z = iszz - cz

        # print('x = {}, y = {}, z = {}'.format(x,y,z))
        img = img[:, x:x + cx, y:y + cy, z:z + cz]
        label = label[:, x:x + cx, y:y + cy, z:z + cz]
        # remove brain label
        # print('after crop: img.shape = {}, range = [{}, {}], label.shape = {}'.format(img.shape, img.min(), img.max(), label.shape))
        # print('before remove brain mask')
        # for i in np.unique(label)[1:]:
        #     print('        label = {}, number of voxels = {}'.format(i, len(np.where(label == i)[0])))
        label[np.nonzero(label)] = label[np.nonzero(label)] - 1
        # print('after remove brain mask')
        # for i in np.unique(label)[1:]:
        #     print('        label = {}, number of voxels = {}'.format(i, len(np.where(label == i)[0])))
        # assert(0)

    else:

        label_max = label.max()
        # print('label_max', label_max)
        idx = np.where(label == label_max)
        ctrx = (idx[1].max() + idx[1].min()) / 2
        ctry = (idx[2].max() + idx[2].min()) / 2
        ctrz = (idx[3].max() + idx[3].min()) / 2

        # print(idx[1].min(), idx[1].max(), ctrx)
        # print(idx[2].min(), idx[2].max(), ctry)
        # print(idx[3].min(), idx[3].max(), ctrz)

        z0 = max(0, ctrz - cz / 2)
        if z0 + cz > iszz: z0 = iszz - cz
        y0 = max(0, ctry - cy / 2)
        if y0 + cy > iszy: y0 = iszy - cy
        x0 = max(0, ctrx - cx / 2)
        if x0 + cx > iszx: x0 = iszx - cx

        # print(x0, y0, z0)
        # assert (0)
        img = img[:, x0:x0 + cx, y0:y0 + cy, z0:z0 + cz]
        label = label[:, x0:x0 + cx, y0:y0 + cy, z0:z0 + cz]

    # assert (img.ndim == label.ndim)
    # assert(0)
    # flip sup-inf
    if p_si > 0 and not istest:
        v = bernoulli.rvs(p_si, size=1)
        if v == 1:
            # print('p_si. img.shape = {}'.format(img.shape))
            # assert(0)
            img = np.flip(img, axis=3).copy()
            if isinstance(imb, tuple):
                label = np.flip(label, axis=3).copy()

    # flip up-down
    if p_ud > 0 and not istest:
        v = bernoulli.rvs(p_ud, size=1)
        if v == 1:
            # print('flip ud...')
            img = np.flip(img, axis=2).copy()
            if isinstance(imb, tuple):
                label = np.flip(label, axis=2).copy()

    # flip left-right
    if p_lr > 0 and not istest:
        v = bernoulli.rvs(p_lr, size=1)
        if v == 1:
            # print('flip lr...')
            img = np.flip(img, axis=1).copy()
            if isinstance(imb, tuple):
                label = np.flip(label, axis=1).copy()

    # img = np.rollaxis(img, 2, 0)
    # print('img.shape = {}'.format(img.shape))
    # for i in range(nch):
    #     print(' final img[{}]: min = {}, max = {}'.format(i, img[i].min(), img[i].max()))

    # assert(0)
    if isinstance(imb, tuple):
        return img, label
    else:
        return img

def img_augument3d_v3(imb, crop, shift=0, rotate=0, scale=0, normalize_pctwise=None, p_si=0, p_lr=0, p_ud=0,
                   contrast=None, gamma=None, valid_list=False, istest=False, mrgn=8):
    """
    default value for normalize_pctwise=(20,95)
    as compared to img_augument3d: scale image voxel values based on the statistics within brain
    AND crop voi before any spatial augumentation
    YY, 06/14/20
    """
    if isinstance(imb, tuple) and len(imb) == 2:
        img, label = imb
    else:
        img = imb
        label = None
    # default input img: (z,y,x,c), range:[0,1]
    # obtain mean and std for each channel, and the brain mask
    # print('raw img: type = {}, shape = {}, min ={}, max = {}'.format(img.dtype, img.shape, img.min(), img.max()))
    # print('raw label:type = {},shape = {}, range = {}'.format(label.dtype, label.shape, np.unique(label)))
    # for i in np.unique(label)[1:]:
    #     print('label = {}, number of voxels = {}'.format(i, len(np.where(label==i)[0])))
    # assert(0)
    nch, iszx, iszy, iszz = img.shape
    brain_mask = np.zeros((iszx, iszy, iszz), dtype=np.uint8)
    ch_stats = np.zeros((nch, 2), dtype=np.float32)
    orig_label_values = np.unique(label)
    ch_max = []
    for i in range(nch):
        brain_mask[np.where(img[i] > 0)] = 1
        # ch_stats[i] = np.asarray([np.mean(img[i]), np.std(img[i])])
        # print('before scale: img: min = {}, max = {}, mean = {}, std = {}'.format(img[i].min(), img[i].max(), np.mean(img[i]), np.std(img[i])))
        # img[i] = (img[i] - np.mean(img[i])) / np.std(img[i])
        ch_max.append(img[i].max())
        img[i] = img[i] / ch_max[i]
        # print('       scaled:img: min = {}, max = {}'.format(img[i].min(), img[i].max()))

    ch_max = np.asarray(ch_max).astype(np.float32)
    # print('ch_max', ch_max)

    # print('ch_stats', ch_stats)
    # print('brain values = {}, size = {}'.format(np.unique(brain_mask), np.count_nonzero(brain_mask)))
    if normalize_pctwise is not None:
        # raise ValueError('normalized_pitwise needs to be rewritten!')
        pclow, pchigh = normalize_pctwise
        for i in range(nch):
            if not istest:
                pclow_i = np.random.randint(pclow[0], pclow[1])
                pchigh_i = np.random.randint(pchigh[0], pchigh[1])
            else:
                pclow_i = pclow
                pchigh_i= pchigh
            img_i = img[i]
            vl, vh = np.percentile(img_i[np.where(brain_mask==1)], (pclow_i, pchigh_i))
            #print('i = {}, before norm: img_i range: [{}, {}], vl[{}] = {}, vh[{}] = {}'.format(i,img_i.min(), img_i.max(),
            #                                                                                    pclow_i, vl,
            #                                                                                    pchigh_i,vh))
            img[i] = exposure.rescale_intensity(img[i], in_range=(vl, vh))  # range: 0 - 1
            #print('        after norm: img_i range: [{}, {}]'.format(img[i].min(), img[i].max()))
    # assert(0)
    # print('raw img.shape = {},min ={}, max = {}'.format(img.shape, img.min(), img.max()))
    # print('raw label range = {}'.format(np.unique(label)))
    if contrast is not None:
        if not istest:
            img = augment_contrast(img, contrast_range=contrast, preserve_range=True, per_channel=True)
    # print('after contrast img.shape = {},min ={}, max = {}'.format(img.shape, img.min(), img.max()))
    # assert(0)
    if gamma is not None:
        img = augment_gamma(img, gamma_range=gamma, invert_image=False, epsilon=1e-7, per_channel=False,
                            retain_stats=False)
    # print('after gamma img.shape = {},min ={}, max = {}'.format(img.shape, img.min(), img.max()))
    # assert(0)
    iz, iy, ix, cz, cy, cx = crop
    if not istest:
        label = label + np.expand_dims(brain_mask, axis=0)
        # print('after add brain mask, label.shape = {}, classes = {}'.format(label.shape, np.unique(label)))
        # for i in np.unique(label)[1:]:
        #     print('        label = {}, number of voxels = {}'.format(i, len(np.where(label == i)[0])))
        # assert(0)
        if rotate > 0 or scale > 0:
            # print rotate, scale
            patch_size = (cx, cy, cz)  # crop[3:] (cx,cy,cz)
            # print('patch_size = {}'.format(patch_size))
            '''
            basic_patch_size = get_patch_size(patch_size,
                                              rot_x = (-15./360 * 2. * np.pi, 15./360 * 2. * np.pi),
                                              rot_y = (-15./360 * 2. * np.pi, 15./360 * 2. * np.pi),
                                              rot_z = (-15./360 * 2. * np.pi, 15./360 * 2. * np.pi),
                                              scale_range = (0.85, 1.15))
            '''
            # print('basic_patch_size', basic_patch_size)
            # assert(0)
            # img = np.swapaxes(img, 1, 3)
            # save the current img range and normalize the img to [0,1]
            # ch_min = np.min(img, axis=(1, 2, 3), keepdims=True)
            # ch_max = np.max(img, axis=(1, 2, 3), keepdims=True)
            # print('ch_min = {}, ch_max = {}'.format(ch_min, ch_max))
            # img = (img - ch_min) / (ch_max - ch_min)

            img = np.expand_dims(img, axis=0)
            # label = np.swapaxes(np.expand_dims(label, axis=0), 1, 3)
            label = np.expand_dims(label, axis=0)

            # print('before spatial augment: img', img.shape, img.min(), img.max())
            # print('       no_of_nonzeros', np.count_nonzero(img))
            # print('       no_of_brain', len(np.where(label>0)[0]))
            # print('before spatial augment: label', label.shape, np.unique(label))
            # print('patch_size', patch_size)

            fake_patch_size = (iszx, iszy, iszz)
            # print('fake_patch_size', fake_patch_size)

            do_rotation = (rotate > 0 )
            do_scale = (scale > 0)
            # print(do_rotation, do_scale)
            img, label = augment_spatial(img, label, fake_patch_size,
                                         patch_center_dist_from_border=(cx / 2, cy / 2, cz / 2),
                                         do_elastic_deform=True,
                                         alpha=(0., 900.), sigma=(9., 13.),
                                         do_rotation=do_rotation,
                                         angle_x=(-rotate / 360. * 2. * np.pi, rotate / 360. * 2. * np.pi),
                                         angle_y=(-rotate / 360. * 2. * np.pi, rotate / 360. * 2. * np.pi),
                                         angle_z=(-rotate / 360. * 2. * np.pi, rotate / 360. * 2. * np.pi),
                                         do_scale=do_scale,
                                         scale=(1.0 - scale, 1.0 + scale),
                                         random_crop=False
                                         )
            # img = img.clip(min=0, max=1.0)
            img = img.squeeze(axis=0)
            label = label.squeeze(axis=0).astype(np.uint8)
            # brain_mask[label>0] = 1
            # label = label - 1
            # print('after spatial augment: img', img.shape, img.min(), img.max())
            # print('    no_of_ge_0 = {}, no_of_ge_1 = {}'.format(len(np.where(img > 0.0)[0]),
            #                                                np.count_nonzero(np.where(img > 1.0))))
            #bg_idx = np.where(label.squeeze() == 0)
            # print('    no_of_bkgrd', len(bg_idx[0]))

            #for i in range(nch):
            #    img[i][bg_idx] = 0

            # print('     no_of_ge_0_clip = {}'.format(len(np.where(img > 0.0)[0])))
            # print('after crop by brain: img', img.shape, img.min(), img.max())
            img = img.clip(min=0.0, max = 1.0)
            # print('after final crop: img', img.shape, img.min(), img.max())
            # print('after spatial augment: label', label.shape, np.unique(label))

            # assert (0)
        # label should have labels as [1,2,3,4] in which 1 is the brain
        # cls = np.asarray([1,2,3,4],dtype=np.uint8)
        cls = np.unique(label)[1:]
        p_cls = 1.0 * cls / cls.sum()
        # print('cls = {}, p_cls = {}'.format(cls, p_cls))
        c_sample = np.random.choice(cls, size=1, p=p_cls)[0]
        # c_sample = 1
        # idx = np.where(label == c_sample)
        # print('c_sample:',c_sample)
        # if cls[-1] < 4:
        #     print(cls, p_cls, c_sample)
        # assert(0)
        # assert(0)
        if c_sample == 1:  # brain
            idx = np.where(label > 0)

            brsx = max(idx[1].max() - idx[1].min() + 1, cx) + 2 * mrgn
            start_x = max(0, idx[1].min() - mrgn)
            end_x = min(iszx, start_x + brsx) - cx
            # assert(end_x >= start_x)
            if end_x < start_x + 2 * mrgn:
                start_x = max(0, end_x - 2 * mrgn)
                # print('idx[1].min = {}, idx[1].max = {}, brsx = {}, start_x = {}, end_x = {}'.format(idx[1].min(), idx[1].max(), brsx, start_x, end_x))
                # assert(0)
            x = np.random.randint(low=start_x, high=end_x, size=1)[0]
            # print('idx[1].min = {}, idx[1].max = {}, brsx = {}, start_x = {}, end_x = {}, x = {}'.format(idx[1].min(), idx[1].max(), brsx, start_x, end_x, x))

            brsy = max(idx[2].max() - idx[2].min() + 1, cy) + 2 * mrgn
            start_y = max(0, idx[2].min() - mrgn)
            end_y = min(iszy, start_y + brsy) - cy
            if end_y < start_y + 2 * mrgn:
                start_y = max(0, end_y - 2 * mrgn)
                # print('idx[2].min = {}, idx[2].max = {}, brsy = {}, start_y = {}, end_y = {}'.format(idx[2].min(), idx[2].max(), brsy, start_y, end_y))
                # assert(0)
            y = np.random.randint(low=start_y, high=end_y, size=1)[0]
            # print('idx[2].min = {}, idx[2].max = {}, brsy = {}, start_y = {}, end_y = {}, y = {}'.format(idx[2].min(), idx[2].max(), brsy, start_y, end_y, y))

            brsz = max(idx[3].max() - idx[3].min() + 1, cz) + 2 * mrgn
            start_z = max(0, idx[3].min() - mrgn)
            end_z = min(iszz, start_z + brsz) - cz
            if end_z < start_z + 2 * mrgn:
                start_z = max(0, end_z - 2 * mrgn)
                # print('idx[3].min = {}, idx[3].max = {}, brsz = {}, start_z = {}, end_z = {}'.format(idx[3].min(), idx[3].max(), brsz, start_z, end_z))
                # assert(0)
            z = np.random.randint(low=start_z, high=end_z, size=1)[0]
            # print('idx[3].min = {}, idx[3].max = {}, brsz = {}, start_z = {}, end_z = {}, z = {}'.format(idx[3].min(), idx[3].max(), brsz, start_z, end_z, z))

        else:
            idx = np.where(label == c_sample)
            n_idx = len(idx[1])
            # print('n_idx = {}'.format(n_idx))
            # assert(n_idx > 0)
            if n_idx == 0:
                print('c_sample = {}, label vlaues = {}, orig_label_values = {}'.format(c_sample, np.unique(label),
                                                                                        orig_label_values))
                assert (0)
            i = np.random.randint(low=0, high=n_idx, size=1)[0]
            ctrx = idx[1][i]
            ctry = idx[2][i]
            ctrz = idx[3][i]
            # print('the {} th voxel[{},{},{}] with label [{}] is selected'.format(i, ctrx, ctry, ctrz, c_sample))
            assert (label[:, ctrx, ctry, ctrz] == c_sample)

            x = max(0, ctrx - cx / 2)
            if x + cx > iszx: x = iszx - cx
            y = max(0, ctry - cy / 2)
            if y + cy > iszy: y = iszy - cy
            z = max(0, ctrz - cz / 2)
            if z + cz > iszz: z = iszz - cz

        # print('x = {}, y = {}, z = {}'.format(x,y,z))
        img = img[:, x:x + cx, y:y + cy, z:z + cz]
        label = label[:, x:x + cx, y:y + cy, z:z + cz]
        # remove brain label
        # print('after crop: img.shape = {}, range = [{}, {}], label.shape = {}'.format(img.shape, img.min(), img.max(),
        #                                                                              label.shape))
        # print('    crop: non_neg = {}, non_ge_1 = {}'.format(np.count_nonzero(np.where(img < 0.0)),
        #                                                    np.count_nonzero(np.where(img > 1.0))))
        # print('before remove brain mask')
        # for i in np.unique(label)[1:]:
        #     print('        label = {}, number of voxels = {}'.format(i, len(np.where(label == i)[0])))
        label[np.nonzero(label)] = label[np.nonzero(label)] - 1
        # print('after remove brain mask')
        # for i in np.unique(label)[1:]:
        #     print('        label = {}, number of voxels = {}'.format(i, len(np.where(label == i)[0])))
        # assert (0)

    else:

        label_max = label.max()
        # print('label_max', label_max)
        idx = np.where(label == label_max)
        ctrx = (idx[1].max() + idx[1].min()) / 2
        ctry = (idx[2].max() + idx[2].min()) / 2
        ctrz = (idx[3].max() + idx[3].min()) / 2

        # print(idx[1].min(), idx[1].max(), ctrx)
        # print(idx[2].min(), idx[2].max(), ctry)
        # print(idx[3].min(), idx[3].max(), ctrz)

        z0 = max(0, ctrz - cz / 2)
        if z0 + cz > iszz: z0 = iszz - cz
        y0 = max(0, ctry - cy / 2)
        if y0 + cy > iszy: y0 = iszy - cy
        x0 = max(0, ctrx - cx / 2)
        if x0 + cx > iszx: x0 = iszx - cx

        # print(x0, y0, z0)
        # assert (0)
        img = img[:, x0:x0 + cx, y0:y0 + cy, z0:z0 + cz]
        label = label[:, x0:x0 + cx, y0:y0 + cy, z0:z0 + cz]

    # assert (img.ndim == label.ndim)
    # assert(0)
    # flip sup-inf
    if p_si > 0 and not istest:
        v = bernoulli.rvs(p_si, size=1)
        if v == 1:
            # print('p_si. img.shape = {}'.format(img.shape))
            # assert(0)
            img = np.flip(img, axis=3).copy()
            if isinstance(imb, tuple):
                label = np.flip(label, axis=3).copy()

    # flip up-down
    if p_ud > 0 and not istest:
        v = bernoulli.rvs(p_ud, size=1)
        if v == 1:
            # print('flip ud...')
            img = np.flip(img, axis=2).copy()
            if isinstance(imb, tuple):
                label = np.flip(label, axis=2).copy()

    # flip left-right
    if p_lr > 0 and not istest:
        v = bernoulli.rvs(p_lr, size=1)
        if v == 1:
            # print('flip lr...')
            img = np.flip(img, axis=1).copy()
            if isinstance(imb, tuple):
                label = np.flip(label, axis=1).copy()

    # img = np.rollaxis(img, 2, 0)
    # print('img.shape = {}'.format(img.shape))
    # for i in range(nch):
    #     print(' final img[{}]: min = {}, max = {}'.format(i, img[i].min(), img[i].max()))

    # assert(0)
    if isinstance(imb, tuple):
        return img, label
    else:
        return img

def img_augument3d_v3_bak200615(imb, crop, shift=0, rotate=0, scale=0, normalize_pctwise=None, p_si=0, p_lr=0, p_ud=0,
                   contrast=None, gamma=None, valid_list=False, istest=False, mrgn=8):
    """
    default value for normalize_pctwise=(20,95)
    as compared to img_augument3d: scale image voxel values based on the statistics within brain
    AND crop voi before any spatial augumentation
    YY, 06/14/20
    """
    if isinstance(imb, tuple) and len(imb) == 2:
        img, label = imb
    else:
        img = imb
        label = None
    # default input img: (z,y,x,c), range:[0,1]
    # obtain mean and std for each channel, and the brain mask
    # print('raw img: type = {}, shape = {}, min ={}, max = {}'.format(img.dtype, img.shape, img.min(), img.max()))
    # print('raw label:type = {},shape = {}, range = {}'.format(label.dtype, label.shape, np.unique(label)))
    # for i in np.unique(label)[1:]:
    #     print('label = {}, number of voxels = {}'.format(i, len(np.where(label==i)[0])))
    # assert(0)
    nch, iszx, iszy, iszz = img.shape
    brain_mask = np.zeros((iszx, iszy, iszz), dtype=np.uint8)
    ch_stats = np.zeros((nch, 2), dtype=np.float32)
    orig_label_values = np.unique(label)
    ch_max = []
    for i in range(nch):
        brain_mask[np.where(img[i] > 0)] = 1
        # ch_stats[i] = np.asarray([np.mean(img[i]), np.std(img[i])])
        # print('before scale: img: min = {}, max = {}, mean = {}, std = {}'.format(img[i].min(), img[i].max(), np.mean(img[i]), np.std(img[i])))
        # img[i] = (img[i] - np.mean(img[i])) / np.std(img[i])
        ch_max.append(img[i].max())
        img[i] = img[i] / ch_max[i]
        # print('       scaled:img: min = {}, max = {}'.format(img[i].min(), img[i].max()))

    ch_max = np.asarray(ch_max).astype(np.float32)
    # print('ch_max', ch_max)

    # print('ch_stats', ch_stats)
    # print('brain values = {}, size = {}'.format(np.unique(brain_mask), np.count_nonzero(brain_mask)))
    if normalize_pctwise is not None:
        # raise ValueError('normalized_pitwise needs to be rewritten!')
        pclow, pchigh = normalize_pctwise
        for i in range(nch):
            if not istest:
                pclow_i = np.random.randint(pclow[0], pclow[1])
                pchigh_i = np.random.randint(pchigh[0], pchigh[1])
            else:
                pclow_i = pclow
                pchigh_i= pchigh
            img_i = img[i]
            vl, vh = np.percentile(img_i[np.where(brain_mask==1)], (pclow_i, pchigh_i))
            #print('i = {}, before norm: img_i range: [{}, {}], vl[{}] = {}, vh[{}] = {}'.format(i,img_i.min(), img_i.max(),
            #                                                                                    pclow_i, vl,
            #                                                                                    pchigh_i,vh))
            img[i] = exposure.rescale_intensity(img[i], in_range=(vl, vh))  # range: 0 - 1
            #print('        after norm: img_i range: [{}, {}]'.format(img[i].min(), img[i].max()))
    # assert(0)
    # print('raw img.shape = {},min ={}, max = {}'.format(img.shape, img.min(), img.max()))
    # print('raw label range = {}'.format(np.unique(label)))
    if contrast is not None:
        img = augment_contrast(img, contrast_range=contrast, preserve_range=True, per_channel=True)
    # print('after contrast img.shape = {},min ={}, max = {}'.format(img.shape, img.min(), img.max()))
    if gamma is not None:
        img = augment_gamma(img, gamma_range=gamma, invert_image=False, epsilon=1e-7, per_channel=False,
                            retain_stats=False)
    # print('after gamma img.shape = {},min ={}, max = {}'.format(img.shape, img.min(), img.max()))
    # assert(0)
    iz, iy, ix, cz, cy, cx = crop
    if not istest:
        label = label + np.expand_dims(brain_mask, axis=0)
        # print('after add brain mask, label.shape = {}, classes = {}'.format(label.shape, np.unique(label)))
        # for i in np.unique(label)[1:]:
        #     print('        label = {}, number of voxels = {}'.format(i, len(np.where(label == i)[0])))
        # assert(0)
        if rotate > 0 and scale > 0:
            # print rotate, scale
            patch_size = (cx, cy, cz)  # crop[3:] (cx,cy,cz)
            # print('patch_size = {}'.format(patch_size))
            '''
            basic_patch_size = get_patch_size(patch_size,
                                              rot_x = (-15./360 * 2. * np.pi, 15./360 * 2. * np.pi),
                                              rot_y = (-15./360 * 2. * np.pi, 15./360 * 2. * np.pi),
                                              rot_z = (-15./360 * 2. * np.pi, 15./360 * 2. * np.pi),
                                              scale_range = (0.85, 1.15))
            '''
            # print('basic_patch_size', basic_patch_size)
            # assert(0)
            # img = np.swapaxes(img, 1, 3)
            # save the current img range and normalize the img to [0,1]
            # ch_min = np.min(img, axis=(1, 2, 3), keepdims=True)
            # ch_max = np.max(img, axis=(1, 2, 3), keepdims=True)
            # print('ch_min = {}, ch_max = {}'.format(ch_min, ch_max))
            # img = (img - ch_min) / (ch_max - ch_min)

            img = np.expand_dims(img, axis=0)
            # label = np.swapaxes(np.expand_dims(label, axis=0), 1, 3)
            label = np.expand_dims(label, axis=0)

            # print('before spatial augment: img', img.shape, img.min(), img.max())
            # print('       no_of_nonzeros', np.count_nonzero(img))
            # print('       no_of_brain', len(np.where(label>0)[0]))
            # print('before spatial augment: label', label.shape, np.unique(label))
            # print('patch_size', patch_size)

            fake_patch_size = (iszx, iszy, iszz)
            # print('fake_patch_size', fake_patch_size)

            img, label = augment_spatial(img, label, fake_patch_size,
                                         patch_center_dist_from_border=(cx / 2, cy / 2, cz / 2),
                                         do_elastic_deform=True,
                                         alpha=(0., 900.), sigma=(9., 13.),
                                         do_rotation=True,
                                         angle_x=(-rotate / 360. * 2. * np.pi, rotate / 360. * 2. * np.pi),
                                         angle_y=(-rotate / 360. * 2. * np.pi, rotate / 360. * 2. * np.pi),
                                         angle_z=(-rotate / 360. * 2. * np.pi, rotate / 360. * 2. * np.pi),
                                         do_scale=True,
                                         scale=(1.0 - scale, 1.0 + scale),
                                         random_crop=False
                                         )
            # img = img.clip(min=0, max=1.0)
            img = img.squeeze(axis=0)
            label = label.squeeze(axis=0).astype(np.uint8)
            # brain_mask[label>0] = 1
            # label = label - 1
            # print('after spatial augment: img', img.shape, img.min(), img.max())
            # print('    no_of_ge_0 = {}, no_of_ge_1 = {}'.format(len(np.where(img > 0.0)[0]),
            #                                                np.count_nonzero(np.where(img > 1.0))))
            bg_idx = np.where(label.squeeze() == 0)
            # print('    no_of_bkgrd', len(bg_idx[0]))

            for i in range(nch):
                img[i][bg_idx] = 0

            # print('     no_of_ge_0_clip = {}'.format(len(np.where(img > 0.0)[0])))
            # print('after crop by brain: img', img.shape, img.min(), img.max())
            img = img.clip(min=0.0, max = 1.0)
            # print('after final crop: img', img.shape, img.min(), img.max())
            # print('after spatial augment: label', label.shape, np.unique(label))

            # assert (0)
        # label should have labels as [1,2,3,4] in which 1 is the brain
        # cls = np.asarray([1,2,3,4],dtype=np.uint8)
        cls = np.unique(label)[1:]
        p_cls = 1.0 * cls / cls.sum()
        # print('cls = {}, p_cls = {}'.format(cls, p_cls))
        c_sample = np.random.choice(cls, size=1, p=p_cls)[0]
        # c_sample = 1
        # idx = np.where(label == c_sample)
        # print('c_sample:',c_sample)
        # if cls[-1] < 4:
        #     print(cls, p_cls, c_sample)
        # assert(0)
        # assert(0)
        if c_sample == 1:  # brain
            idx = np.where(label > 0)

            brsx = max(idx[1].max() - idx[1].min() + 1, cx) + 2 * mrgn
            start_x = max(0, idx[1].min() - mrgn)
            end_x = min(iszx, start_x + brsx) - cx
            # assert(end_x >= start_x)
            if end_x < start_x + 2 * mrgn:
                start_x = max(0, end_x - 2 * mrgn)
                # print('idx[1].min = {}, idx[1].max = {}, brsx = {}, start_x = {}, end_x = {}'.format(idx[1].min(), idx[1].max(), brsx, start_x, end_x))
                # assert(0)
            x = np.random.randint(low=start_x, high=end_x, size=1)[0]
            # print('idx[1].min = {}, idx[1].max = {}, brsx = {}, start_x = {}, end_x = {}, x = {}'.format(idx[1].min(), idx[1].max(), brsx, start_x, end_x, x))

            brsy = max(idx[2].max() - idx[2].min() + 1, cy) + 2 * mrgn
            start_y = max(0, idx[2].min() - mrgn)
            end_y = min(iszy, start_y + brsy) - cy
            if end_y < start_y + 2 * mrgn:
                start_y = max(0, end_y - 2 * mrgn)
                # print('idx[2].min = {}, idx[2].max = {}, brsy = {}, start_y = {}, end_y = {}'.format(idx[2].min(), idx[2].max(), brsy, start_y, end_y))
                # assert(0)
            y = np.random.randint(low=start_y, high=end_y, size=1)[0]
            # print('idx[2].min = {}, idx[2].max = {}, brsy = {}, start_y = {}, end_y = {}, y = {}'.format(idx[2].min(), idx[2].max(), brsy, start_y, end_y, y))

            brsz = max(idx[3].max() - idx[3].min() + 1, cz) + 2 * mrgn
            start_z = max(0, idx[3].min() - mrgn)
            end_z = min(iszz, start_z + brsz) - cz
            if end_z < start_z + 2 * mrgn:
                start_z = max(0, end_z - 2 * mrgn)
                # print('idx[3].min = {}, idx[3].max = {}, brsz = {}, start_z = {}, end_z = {}'.format(idx[3].min(), idx[3].max(), brsz, start_z, end_z))
                # assert(0)
            z = np.random.randint(low=start_z, high=end_z, size=1)[0]
            # print('idx[3].min = {}, idx[3].max = {}, brsz = {}, start_z = {}, end_z = {}, z = {}'.format(idx[3].min(), idx[3].max(), brsz, start_z, end_z, z))

        else:
            idx = np.where(label == c_sample)
            n_idx = len(idx[1])
            # print('n_idx = {}'.format(n_idx))
            # assert(n_idx > 0)
            if n_idx == 0:
                print('c_sample = {}, label vlaues = {}, orig_label_values = {}'.format(c_sample, np.unique(label),
                                                                                        orig_label_values))
                assert (0)
            i = np.random.randint(low=0, high=n_idx, size=1)[0]
            ctrx = idx[1][i]
            ctry = idx[2][i]
            ctrz = idx[3][i]
            # print('the {} th voxel[{},{},{}] with label [{}] is selected'.format(i, ctrx, ctry, ctrz, c_sample))
            assert (label[:, ctrx, ctry, ctrz] == c_sample)

            x = max(0, ctrx - cx / 2)
            if x + cx > iszx: x = iszx - cx
            y = max(0, ctry - cy / 2)
            if y + cy > iszy: y = iszy - cy
            z = max(0, ctrz - cz / 2)
            if z + cz > iszz: z = iszz - cz

        # print('x = {}, y = {}, z = {}'.format(x,y,z))
        img = img[:, x:x + cx, y:y + cy, z:z + cz]
        label = label[:, x:x + cx, y:y + cy, z:z + cz]
        # remove brain label
        # print('after crop: img.shape = {}, range = [{}, {}], label.shape = {}'.format(img.shape, img.min(), img.max(),
        #                                                                              label.shape))
        # print('    crop: non_neg = {}, non_ge_1 = {}'.format(np.count_nonzero(np.where(img < 0.0)),
        #                                                    np.count_nonzero(np.where(img > 1.0))))
        # print('before remove brain mask')
        # for i in np.unique(label)[1:]:
        #     print('        label = {}, number of voxels = {}'.format(i, len(np.where(label == i)[0])))
        label[np.nonzero(label)] = label[np.nonzero(label)] - 1
        # print('after remove brain mask')
        # for i in np.unique(label)[1:]:
        #     print('        label = {}, number of voxels = {}'.format(i, len(np.where(label == i)[0])))
        # assert (0)

    else:

        label_max = label.max()
        # print('label_max', label_max)
        idx = np.where(label == label_max)
        ctrx = (idx[1].max() + idx[1].min()) / 2
        ctry = (idx[2].max() + idx[2].min()) / 2
        ctrz = (idx[3].max() + idx[3].min()) / 2

        # print(idx[1].min(), idx[1].max(), ctrx)
        # print(idx[2].min(), idx[2].max(), ctry)
        # print(idx[3].min(), idx[3].max(), ctrz)

        z0 = max(0, ctrz - cz / 2)
        if z0 + cz > iszz: z0 = iszz - cz
        y0 = max(0, ctry - cy / 2)
        if y0 + cy > iszy: y0 = iszy - cy
        x0 = max(0, ctrx - cx / 2)
        if x0 + cx > iszx: x0 = iszx - cx

        # print(x0, y0, z0)
        # assert (0)
        img = img[:, x0:x0 + cx, y0:y0 + cy, z0:z0 + cz]
        label = label[:, x0:x0 + cx, y0:y0 + cy, z0:z0 + cz]

    # assert (img.ndim == label.ndim)
    # assert(0)
    # flip sup-inf
    if p_si > 0 and not istest:
        v = bernoulli.rvs(p_si, size=1)
        if v == 1:
            # print('p_si. img.shape = {}'.format(img.shape))
            # assert(0)
            img = np.flip(img, axis=3).copy()
            if isinstance(imb, tuple):
                label = np.flip(label, axis=3).copy()

    # flip up-down
    if p_ud > 0 and not istest:
        v = bernoulli.rvs(p_ud, size=1)
        if v == 1:
            # print('flip ud...')
            img = np.flip(img, axis=2).copy()
            if isinstance(imb, tuple):
                label = np.flip(label, axis=2).copy()

    # flip left-right
    if p_lr > 0 and not istest:
        v = bernoulli.rvs(p_lr, size=1)
        if v == 1:
            # print('flip lr...')
            img = np.flip(img, axis=1).copy()
            if isinstance(imb, tuple):
                label = np.flip(label, axis=1).copy()

    # img = np.rollaxis(img, 2, 0)
    # print('img.shape = {}'.format(img.shape))
    # for i in range(nch):
    #     print(' final img[{}]: min = {}, max = {}'.format(i, img[i].min(), img[i].max()))

    # assert(0)
    if isinstance(imb, tuple):
        return img, label
    else:
        return img

def overlap_similarities(pred, reference, label=1):

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    pred = pred == label
    overlap_measures_filter.Execute(reference, pred)

    similarites = {}
    similarites['dice'] = overlap_measures_filter.GetDiceCoefficient()
    similarites['jaccard'] = overlap_measures_filter.GetJaccardCoefficient()
    similarites['voe'] = 1.0 - similarites['jaccard']
    similarites['rvd'] = 1.0 - overlap_measures_filter.GetVolumeSimilarity()

    return similarites


def pt_pre_process(pt, s, c, model='train', save_preprocessed=False):

    assert(model == 'train' or model == 'test')
    # print('*** pre_process for patient {} ***'.format(pt))

    img_dir = s['data_dir'] + '/' + pt

    input_data = []

    for ich in c['img_ch']:
        img_file = os.path.join(img_dir, pt + '_' + ich + '.nii.gz')
        img = sitk.ReadImage(img_file)
        # print(' loaded ' + img_file)
        img_data = sitk.GetArrayFromImage(img)
        # print(img_data.shape)
        # non0 = img_data[np.nonzero(img_data)]
        # print('   {}:  img_data.min = {}, max = {}'.format(ich, img_data.min(), img_data.max()))
        # print('        non zero voxel value [{}, {}]: mean = {:.3f}, std = {:.3f}'.format(non0.min(), non0.max(),
        #                                                                                   non0.mean(), np.std(non0)))
        if img_data.min() < 0:
            neg_idx = np.where(img_data<0)
            #print('        : number of negative values: {}'.format(np.count_nonzero(neg_idx)))
            img_data[neg_idx] *= -1
            # print(img_data[neg_idx].min(), img_data[neg_idx].max())
            # assert(0)

        # img_data = (img_data - non0.mean()) / np.std(non0)
        # print('        img_data range changed: [0, {}] -> [{:.3f}, {:.3f}]'.format(dmax, img_data.min(), img_data.max()))

        input_data.append(np.expand_dims(img_data, axis=3))
    input_data = np.expand_dims(np.concatenate(input_data,axis=3).astype(np.float32), axis=0)
    # print('  input_data.shape = {}'.format(input_data.shape))

    if model == 'train':
        seg_file = os.path.join(img_dir, pt + '_seg.nii.gz')
        seg = sitk.ReadImage(seg_file)
        seg_data = sitk.GetArrayFromImage(seg)
        # print('  original labels: {}'.format(np.unique(seg_data)))
        # original labels:  1 = NCR & NET (necrotic & non-enhancing);
        #                   2 = ED (peritumoral edema);
        #                   4 = ET (enhanced tumor);
        # after relabeling: 1 = WT (whole tumor), any voxels that greater than 0 (= 1 + 4) + 2;
        #                   2 = TC (tumor core), include ET + NCR + NET: = 1 + 4
        #                   3 = ET (enhanced tumor) = 4
        label_data = np.zeros_like(seg_data).astype(np.uint8)
        label_data[np.where(seg_data > 0)]  = 1
        label_data[np.where(seg_data == 1)] = 2
        label_data[np.where(seg_data == 4)] = 3

        if save_preprocessed:
            preprocessed_dir = s['data_root'] + 'preprocessed'
            if not os.path.exists(preprocessed_dir): os.makedirs(preprocessed_dir)
            label = sitk.GetImageFromArray(label_data)
            label.CopyInformation(seg)
            label_file = os.path.join(preprocessed_dir,pt + '_label.nii.gz')
            save_nrrd(label, label_file)
        label_data = np.expand_dims(label_data,axis=0)
        return input_data, label_data
    else:
        return input_data

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
        print('display and save debug info')
        handler.setLevel(logging.INFO)
    else:
        print('display debug info only, no log saved')
        handler.setLevel(logging.ERROR)

    logger.addHandler(handler)

    return logger


class CVEnsembledModels(nn.Module):
    def __init__(self, models, weights):
        super(CVEnsembledModels, self).__init__()
        self.n_models = len(models)

        assert(self.n_models == weights.shape[0])
        assert(self.n_models > 0)
        assert(weights.sum() == 1.0)

        self.models = nn.ModuleList(models)
        self.weights = weights

    def forward(self, x):

        out = 0
        for i in range(self.n_models):
            out_i = self.models[i](x)
            if isinstance(out_i, list):
                out_i = out_i[0]
            out += out_i * self.weights[i]

        return out

def weights_init_xavier(m):
    ''' Initializes m.weight tensors with normal dist (Xavier algorithm)'''

    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)):
        #print('init conv')
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
             m.bias = nn.init.constant_(m.bias, 0.0)
        # nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
        #print('init batchnorm')
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.GroupNorm):
        #print('init groupnorm')
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

def init_weights(net, init_type='xavier', nonlin='relu'):
    print('initialization method [{}]'.format(init_type))
    if init_type == 'xavier':
        net.apply(weights_init_xavier)
    else:
        raise NotImplementedError(
            'initialization method [{}] is not implemented'.format(init_type))

def reset_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)):
        m.reset_parameters()

def print_stdout_to_file(txt, fname, mode='a'):
    with open(fname, mode) as stdout_:
        stdout_.write(txt+'\n')
    print(txt)

def sliding_window_inference(data, voi, net, patch_size, ncls, step_ratio=0.5, importance_map=1.0,
                             nonlin=lambda x:x, ckpt_list=None, verbose=True):

    assert len(data.shape) == 4, "data must be (size_c, size_z,size_y, size_x)"
    step = patch_size * step_ratio
    zyx0 = [] # starting points of crop_voi

    for dim in range(3):
        zyx0.append(np.arange(start=voi[dim], stop=voi[dim+3]+voi[dim]-step[dim], step=step[dim], dtype=np.int16))
        if zyx0[dim][-1] + patch_size[dim] > voi[dim+3] + voi[dim]:
            zyx0[dim][-1] = voi[dim+3] + voi[dim] - patch_size[dim] - 1
    #print('zyx0: ',zyx0)
    zyx = np.array(np.meshgrid(zyx0[0], zyx0[1], zyx0[2])).T.reshape(-1, 3).astype(np.int16)

    if verbose:
        print('data.shape: ', data.shape)
        print('voi: ', voi)
        print('step: ', step)
        print('ncls: ', ncls)
        print('n_zyx: ', zyx.shape[0])
        print('zyx0: ', zyx0)
    # assert(0)
    voi_output = []
    if ckpt_list is not None:
        i = 1
        for check_point in ckpt_list:
            net.load_state_dict(check_point)
            print(' loaded No. {} check_point'.format(i))
            patch_output = patch_inference(data, voi, net, patch_size,zyx, ncls, importance_map, nonlin=nonlin, verbose=verbose)
            voi_output.append(patch_output.cpu().numpy())
            i = i + 1
            # voi_output.append(patch_output)
        voi_output = np.asarray(voi_output).astype(np.float32)
        voi_output = np.mean(voi_output, axis=0) # return np.array
    else:
        # return a tensor in gpu (for validation use)
        voi_output = patch_inference(data, voi, net, patch_size, zyx, ncls, importance_map,nonlin=nonlin, verbose=verbose)

    return voi_output

def patch_inference(data, voi, net, patch_size, zyx, ncls, importance_map=1.0, nonlin=lambda x:x, verbose=True):
    '''
    performce inference for each patch and fill the results to the size of voi[3:]
    data: original data as tensor
    net: model
    patch_size: the size of each patch in [patch_size_z, patch_size_y, patch_size_x]
    zyx: the coordiante of the staring point of each patch: [z,y,x]
    ncls: number of classes
    importance_map:
    '''
    output = torch.zeros((ncls, voi[3], voi[4], voi[5]), dtype=torch.float32, requires_grad=False).cuda(non_blocking=True)
    n_predictions = torch.zeros_like(output, dtype=torch.half, requires_grad=False).cuda(non_blocking=True)

    pz, py, px = patch_size
    if verbose: print('output.shape ', output.shape, ', patch_size: ', patch_size)

    for k, (z, y, x) in enumerate(zyx):
        data_k = torch.unsqueeze(data[:,z:z+pz, y:y+py, x:x+px], dim=0) # b,c,z,y,x
        if verbose: print('patch {}: {}'.format(k, (z, y, x)))
        with torch.no_grad():
            output_k = net(data_k)
            if isinstance(output_k, list): output_k = output_k[0]
            #assert(0)
            output_k = nonlin(output_k).squeeze(dim=0)
            # print('output_k ', output_k.size())
            # output_k = output_k.cpu().numpy().squeeze()
            output_k[:,:] *= importance_map
            z = z - voi[0]
            y = y - voi[1]
            x = x - voi[2]
            # print('z,y,x', z,y,x)
            output[:,z:z+pz,y:y+py,x:x+px] += output_k
            n_predictions[:,z:z+pz,y:y+py,x:x+px] += importance_map

    # output = np.nanmean(output, axis=0)
    # print('n_prediction: ', torch.unique(n_predictions))
    output = output / n_predictions

    return output