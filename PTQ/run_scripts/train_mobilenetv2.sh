#!/usr/bin/env bash

## mobilenetv2 1.5Mb
#python main_imagenet.py --data_path /home/data/imagenet/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3]"

# mobilenetv2 1.3Mb
# python main_imagenet.py --data_path /home/data/imagenet/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 2]"

## mobilenetv2 1.1Mb
#python main_imagenet.py --data_path /home/data/imagenet/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 2, 2, 2]"
#
## mobilenetv2 0.9Mb
#python main_imagenet.py --data_path /home/data/imagenet/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 3, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2]"

#"""CDP"""
## mobilenetv2 1.5Mb cdp
# python main_imagenet.py --data_path /home/data/imagenet/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3]" --gpu '0'

# mobilenetv2 1.3Mb cdp
# python main_imagenet.py --data_path /home/data/imagenet/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2]" --gpu '1'

## mobilenetv2 1.1Mb cdp
# python main_imagenet.py --data_path /home/data/imagenet/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 2, 2, 2]" --gpu '3'
#
## mobilenetv2 0.9Mb cdp
# python main_imagenet.py --data_path /home/data/imagenet/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 2, 2, 4, 3, 2, 4, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2]" --gpu '3'

#"""CDP加上beta后"""
## mobilenetv2 1.5Mb cdp 修改前 前后不变
python main_imagenet.py --data_path /home/data/imagenet/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3]" --gpu '0'

# mobilenetv2 1.3Mb cdp 修改前
# python main_imagenet.py --data_path /home/data/imagenet/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2]" --gpu '1'

## mobilenetv2 1.1Mb cdp 修改前
# python main_imagenet.py --data_path /home/data/imagenet/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 2, 3, 4, 2, 2, 2]" --gpu '1'
#
## mobilenetv2 0.9Mb cdp 修改前
# python main_imagenet.py --data_path /home/data/imagenet/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 3, 4, 4, 3, 4, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2]" --gpu '3'

