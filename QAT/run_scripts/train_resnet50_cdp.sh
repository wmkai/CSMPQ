#!/usr/bin/env bash
python quant_train.py \
 -a resnet50 \
 --epochs 90 \
 --lr 0.0001 \
 --batch_size 1024 \
 --data /home/data/imagenet/ \
 --save_path "saved_quant_model/train_resnet50_${RANDOM}" \
 --act_range_momentum=0.99 \
 --wd 1e-4 \
 --data_percentage 1 \
 --pretrained \
 --fix_BN \
 --checkpoint_iter -1 \
 --gpu_id '0,1' \
 --quant_scheme cdp_modelsize_15.9_a5_143BOP