#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract_cdp.py \
 --model "resnet50" \
 --path "Exp_base/resnet50_base" \
 --dataset "imagenet" \
 --save_path '/home/data/imagenet' \
 --beta 3.3 \
 --model_size 16.0 \
 --quant_type "QAT"