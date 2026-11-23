#!/bin/bash

output_root=${1:-/mnt/data/brats2021/demo_outputs}

# run from the code directory, as the configs/*.csv paths are relative
cd "$(dirname "$(readlink -f "$0")")" || exit 1

mkdir -p ${output_root}/nohup

nohup python -u -W ignore gcml_brats_train.py -f 0 -c config_vFL --gpu all --batch-size 1 --epochs 200 --valid-idx 1 --seed 2025 -l --setting SETTINGS_tr --reset --eval_post --gmlavg --dml_pair=4 --output_dir ${output_root}/200ep/ \
  > ${output_root}/nohup/demo_gcml_200ep.out 2>&1 &
echo $! > ${output_root}/nohup/demo_gcml_200ep.pid
echo "started, pid $(cat ${output_root}/nohup/demo_gcml_200ep.pid), output: ${output_root}/nohup/demo_gcml_200ep.out"