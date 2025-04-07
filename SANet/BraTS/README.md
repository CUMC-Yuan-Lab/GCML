# Demo code for SA-Net model on BraTS21 dataset

## 1. Create Conda Environment
```bash
conda config --set channel_priority flexible
conda env create -f env/brats_environment.yml
conda activate brats
```

## 2. Set input data path
```bash
export INPUT_BRATS_DATA=/Path/To/Your/BraTS21/Data/
```
e.g. ```export INPUT_BRATS_DATA=/data/lab/shared_data/BraTS21/BraTS2021_TrainingData/```

## 3. Train
### 3.1 To train on all BraTS21 data (as in [1]), run:
```bash
source brats_train_all_sites.sh
```
Results will be generated under ```outputs_train/config_vbrats_8_sites/```, including trained model and log. <br>

### 3.2 Alternatively, to train on data from 8 specific BraTS21 sites (as in [2]), run:
```bash
source brats_train_8_sites.sh
```
Results will be generated under ```outputs_train/config_vbrats_all_sites/```, including trained model and log.

## 4. Test
### 4.1 If trained on all BraTS21 data (as in 3.1), run:
```bash
source brats_predict_all_sites.sh model_type
```
where model_type can be ```best_val_loss```,```best_val_loss_ma``` or ```history```<br>
e.g. ```source brats_predict_all_sites.sh best_val_loss``` <br>
Results will be generated under ```outputs_predict/config_vbrats_all_sites/```, including segmentation masks and metrics sheet.<br>

### 4.2 If trained on data from 8 specific BraTS21 sites (as in 3.2), run:
```bash
source brats_predict_8_sites.sh model_type
```
Results will be generated under ```outputs_predict/config_vbrats_8_sites/```, including segmentation masks and metrics sheet.<br>

## Citation
If you use this code for your research, please cite the following papers:
```bibtex
@inbook{Yuan2022,
   author = {Yading Yuan},
   doi = {10.1007/978-3-031-09002-8_4},
   booktitle = {International MICCAI Brainlesion workshop},
   pages = {42-53},
   title = {Evaluating Scale Attention Network for Automatic Brain Tumor Segmentation with Large Multi-parametric MRI Database},
   year = {2022}
}
@article{Chen2025,
   author = {Jingyun Chen and Yading Yuan},
   doi = {10.1109/TMI.2025.3549292},
   issn = {0278-0062},
   journal = {IEEE Transactions on Medical Imaging},
   pages = {1-1},
   title = {Decentralized Personalization for Federated Medical Image Segmentation via Gossip Contrastive Mutual Learning},
   year = {2025}
}
```

## References
[1] Y. Yuan, “Evaluating Scale Attention Network for Automatic Brain Tumor Segmentation with Large Multi-parametric MRI 
Database,” in International MICCAI Brainlesion workshop, 2022, pp. 42–53. doi: 10.1007/978-3-031-09002-8_4. <br>
[2] J. Chen and Y. Yuan, "Decentralized Personalization for Federated Medical Image Segmentation via Gossip Contrastive 
Mutual Learning," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2025.3549292.