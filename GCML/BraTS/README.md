# Demo Code for GCML with SA-Net on BraTS21 dataset

Training and testing of GCML (gossip learning with mutual learning between an incoming and a local model) for brain
tumor segmentation (WT, TC, ET) on the BraTS 2021 data, split into 8 sites. Only the code needed for GCML is included.

## 1. Create Conda Environment
Run from the folder of this README.md:
```bash
conda env create -f env/brats_environment.yml
conda activate brats
```
A CUDA GPU is required.

## 2. Data
The BraTS21 data can be downloaded from Kaggle:
https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1.

This demo simulates federated learning on 8 sites (ID: `4, 5, 6, 7, 11, 16, 20, 21`) with 227 BraTS21 cases, each case belonging to exactly one site. The site
information of the cases is given by the case lists in `configs/`:
- `configs/segtraining21-site-N.csv` lists the cases (`BraTS2021_xxxxx`) of site `N`.
- `configs/config-site-N.py` splits the list of site `N` by row numbers into training (`index_train`), validating (`index_valid`) and testing (`index_test`) cases.

Set `data_dir` in `settings/SETTINGS_tr.py` to the uncompressed data folder (default: ```/mnt/data/brats2021/BraTS2021_Training_Data_full/```).

## 3. Train
Run from the folder of this README.md:
```bash
bash gcml_brats_train.sh [output_root] # default: /mnt/data/brats2021/demo_outputs
```

## 4. Test
Run from the folder of this README.md:
```bash
bash gcml_brats_predict.sh [output_root] # default: /mnt/data/brats2021/demo_outputs
```

## Citation
If you use this code for your research, please cite the following BibTex entries:
```bibtex
@inbook{Yuan2022,
   author = {Yading Yuan},
   doi = {10.1007/978-3-031-09002-8_4},
   booktitle = {International MICCAI Brainlesion workshop},
   pages = {42-53},
   title = {Evaluating Scale Attention Network for Automatic Brain Tumor Segmentation with Large Multi-parametric MRI Database},
   year = {2022}
}
@article{Chen2025,
   author = {Jingyun Chen and Yading Yuan},
   doi = {10.1109/TMI.2025.3549292},
   issn = {0278-0062},
   journal = {IEEE Transactions on Medical Imaging},
   title = {Decentralized Personalization for Federated Medical Image Segmentation via Gossip Contrastive Mutual Learning},
   year = {2025}
}
```
