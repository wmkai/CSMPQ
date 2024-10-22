#!/usr/bin/env bash

## resnet-18 3.0Mb
#python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 3, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2]"
#
## resnet-18 3.5Mb
#python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 3, 2, 2, 2, 3]"
#
## resnet-18 4.0Mb
#python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 4, 3, 2, 3, 3, 3]"

# resnet-18 4.5Mb
# python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 3, 3, 3, 3]"
#
## resnet-18 5.0Mb
# python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 3, 3, 3, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 3, 4, 4, 4]"
#
## resnet-18 5.5Mb
#python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4]"

# CDP

## resnet-18 3.0Mb cdp
# python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 3, 4, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2]" --gpu '0'
# 
## resnet-18 3.5Mb cdp
# python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 4]" --gpu '0'
#
## resnet-18 4.0Mb cdp
# python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 3, 4, 2, 2, 3, 3]" --gpu '1'
# 
## resnet-18 4.5Mb cdp
# python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 3, 4]" --gpu '1'
# 
## resnet-18 5.0Mb cdp
# python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4]" --gpu '3'
#
### resnet-18 5.5Mb cdp
python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4]" --gpu '3'

# CDP修改后 beta 3.3

## resnet-18 3.0Mb cdp
# python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2]" --gpu '0'
# 
## resnet-18 3.5Mb cdp
# python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 2, 2, 2, 2, 4]" --gpu '0'
#
## resnet-18 4.0Mb cdp
# python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4]" --gpu '1'
# 
## resnet-18 4.5Mb cdp
# python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 3, 4]" --gpu '1'
# 
## resnet-18 5.0Mb cdp
# python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4]" --gpu '3'
#
### resnet-18 5.5Mb cdp
python main_imagenet.py --data_path /home/data/imagenet --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4]" --gpu '3'