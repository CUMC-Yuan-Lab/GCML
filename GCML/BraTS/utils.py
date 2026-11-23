import os, logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk

from skimage import exposure
from scipy.stats import bernoulli
from batchgenerators.augmentations.spatial_transformations import augment_spatial
from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_gamma, augment_brightness_additive


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


def weights_init_normal(m):
    ''' Initializes m.weight tensors with normal dist'''

    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, gain=0.02)
        # nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Conv3d):
        nn.init.normal_(m.weight.data, gain=0.02)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, gain=0.02)
        # nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.GroupNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

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

def weights_init_kaiming(m, nonlin='relu'):
    assert(nonlin=='relu' or nonlin == 'leaky_relu')
    # print('kaiming: uniform')
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=nonlin)
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity=nonlin)
        if m.bias is not None:
            m.bias = nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def weights_init_orthogonal(m):

    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal(m.weight.data, gain=0.02)
        # nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.orthogonal(m.weight.data, gain=0.02)
    elif isinstance(m, nn.Linear):
        nn.init.orthogonal(m.weight.data, gain=0.02)
        # nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='xavier', nonlin='relu'):
    print('initialization method [{}]'.format(init_type))
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{}] is not implemented'.format(init_type))

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def print_stdout_to_file(txt, fname, mode='a'):
    with open(fname, mode) as stdout_:
        stdout_.write(txt+'\n')
    print(txt)


############################################ loss functions ############################################

def get_loss(loss_name, dp = 0):
    '''
    return loss function
    :param loss_name: bja_orl (BinaryJaccardLoss) or bja_bfl_orl (BinaryJaccardLoss + BinaryFocalLoss)
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
    else:
        raise ValueError('dp = {} is not implemented, currently support dp = [0,1]'.format(dp))

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


############################################ image processing ############################################

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


############################################ data pre-processing ############################################

def pt_pre_process(pt, s, c, model='train'):

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

        label_data = np.expand_dims(label_data,axis=0)
        return input_data, label_data
    else:
        return input_data
