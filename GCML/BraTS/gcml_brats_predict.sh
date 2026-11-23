#!/bin/bash

config_suffix=FL
output_root=${1:-/mnt/data/brats2021/demo_outputs}
output_dir=${output_root%/}/200ep/
extra_args="${@:2}"

# run from the code directory, as the test case list (configs/segtraining21-te-real8.csv) is a relative path
cd "$(dirname "$(readlink -f "$0")")" || exit 1

pred_dir=${output_dir%/}/predictions/config_v${config_suffix}

# node:first last test case (1-indexed, inclusive) in configs/segtraining21-te-real8.csv
nodes=( "4:1 10"
        "5:11 14"
        "6:15 21"
        "7:22 24"
        "11:25 27"
        "16:28 33"
        "20:34 40"
        "21:41 48" )

# each node's local model on its own test cases
for n in "${nodes[@]}"
do
  node=${n%%:*}
  dice_csv=${pred_dir}/${node}_best_val_loss_f0/dice_tag${config_suffix}_t0.5_f0.csv
  if [ -f "${dice_csv}" ]; then
    echo "node ${node}: Dice result exists, evaluation skipped"
    continue
  fi
  echo "node ${node}, cases ${n##*:}"
  python -u -W ignore gcml_brats_predict.py ${n##*:} -f 0 -c config_v${config_suffix} --setting SETTINGS_te \
    --node ${node} --output_dir ${output_dir} -l ${extra_args} || exit 1
done

# Display per-site mean Dice
python - "${pred_dir}" "${config_suffix}" "${nodes[@]%%:*}" <<'EOF'
import sys, os
import numpy as np

pred_dir, suffix, nodes = sys.argv[1], sys.argv[2], sys.argv[3:]
print('\nPer-site mean Dice (Avg = weighted by each site\'s number of test cases):')
print('{:>6} {:>8}'.format('Site', 'Dice'))
allrows = []
for rank, node in enumerate(nodes, start=1):  # site column: rank order of the site (1st site = 1, ...)
    f = os.path.join(pred_dir, node + '_best_val_loss_f0', 'dice_tag{}_t0.5_f0.csv'.format(suffix))
    if not os.path.isfile(f):
        print('{:>6} {:>8}'.format(rank, 'missing'))
        continue
    dice = np.atleast_2d(np.genfromtxt(f, delimiter=',', usecols=(1, 2, 3)))  # columns: case, WT, TC, ET
    allrows.append(dice)
    print('{:>6} {:8.4f}'.format(rank, np.nanmean(dice)))
if allrows:
    dice = np.vstack(allrows)
    print('{:>6} {:8.4f}'.format('Avg', np.nanmean(dice)))
EOF
