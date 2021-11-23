#!/usr/bin/env bash
python quant_train.py \
 -a resnet18 \
 --epochs 90 \
 --lr 0.0001 \
 --batch_size 1024 \
 --data /home/data/imagenet/ \
 --save_path "saved_quant_model/train_resnet18_${RANDOM}/" \
 --act_range_momentum=0.99 \
 --wd 1e-4 \
 --data_percentage 1 \
 --pretrained \
 --fix_BN \
 --checkpoint_iter -1 \
 --gpu_id '0' \
 --quant_scheme cdp_modelsize_6.7_a6_63B