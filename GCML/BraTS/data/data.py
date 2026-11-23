from torch.utils.data import Dataset
import numpy as np
import pandas as pd


from utils import pt_pre_process, img_augument3d, voi_histogram_equalization


class myFLDatasetSeg3D(Dataset): # JC modified on 23/08/13, for brats_fl
    def __init__(self, s, c, phase='train', folds=None, n_classes=1):

        super(myFLDatasetSeg3D, self).__init__()
        assert (phase == 'train' or phase == 'test')

        self.c = c
        self.s = s
        full_list = self.c['fname_list']
        df = pd.DataFrame(pd.read_csv(full_list, header=None))
        ptnames = [df.values[i][0] for i in range(len(df.values))]
        npt = len(ptnames)
        n_cv = folds[0]
        fold_n = npt // n_cv

        f = folds[1]

        if len(self.c["index_train"]) == 4:
            tr_ptnames = ptnames[self.c["index_train"][0]-1:self.c["index_train"][1]] + ptnames[self.c["index_train"][2]-1:self.c["index_train"][3]]

        else:
            tr_ptnames = ptnames[self.c["index_train"][0] - 1:self.c["index_train"][1]]

        if len(self.c["index_valid"]) == 4:
            te_ptnames = ptnames[self.c["index_valid"][0]-1:self.c["index_valid"][1]] + ptnames[self.c["index_valid"][2]-1:self.c["index_valid"][3]]
        else:
            te_ptnames = ptnames[self.c["index_valid"][0]-1:self.c["index_valid"][1]]

        crop = (c['img_szz'] - c['crop_szz'], c['img_szy'] - c['crop_szy'], c['img_szx'] - c['crop_szx'],
                c['crop_szz'], c['crop_szy'], c['crop_szx'])

        if phase == 'train':
            self.ptnames = tr_ptnames
            print('tr_ptnames = {}'.format(self.c["index_train"]))
            if c['augument_fcn'] == 'img_augument3d':
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
            print('te_ptnames = {}'.format(self.c["index_valid"]))
            if c['augument_fcn'] == 'img_augument3d':
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
