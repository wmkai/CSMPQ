#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract_cdp.py \
 --model "resnet18" \
 --path "Exp_base/resnet18_base" \
 --dataset "imagenet" \
 --save_path '/home/data/imagenet' \
 --beta 1.0 \
 --model_size 5.5 \
 --quant_type "PTQ"
