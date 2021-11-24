#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract_cdp.py \
 --model "mobilenetv2" \
 --path "Exp_base/mobilenetv2_base" \
 --dataset "imagenet" \
 --save_path '/home/data/imagenet/' \
 --beta 3.3 \
 --model_size 1.5 \
 --quant_type "PTQ"