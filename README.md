# CSMPQ
<p>
  <a href="https://img.shields.io/badge/Python-%3E%3D3.7-blue"><img src="https://img.shields.io/badge/Python-%3E%3D3.7-blue"></a>
  <a href="https://img.shields.io/badge/PyTorch-1.9-informational"><img src="https://img.shields.io/badge/PyTorch-1.9-informational"></a>
  <a href="https://img.shields.io/badge/License-MIT-brightgreen"><img src="https://img.shields.io/badge/License-MIT-brightgreen"></a>
</p>
The official implementation of CSMPQ: Class Separability Based Mixed-Precision Quantization.

# Requirements

* [DALI](https://github.com/NVIDIA/DALI) (for accelerating data processing)
* [Apex](https://github.com/NVIDIA/apex) (for distributed running)
* other requirements, running requirements.txt

```python
pip install -r requirements.txt
```



# Running




<font size=6>**Bit Configuration**</font>



```python
#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract_cdp.py \
 --model "resnet18" \
 --path "/Path/to/Base_model" \      # pretrained base model
 --dataset "imagenet" \
 --save_path '/Path/to/Dataset/' \   # Dataset path
 --beta 10.0 \                       # Hyper-parameter for bit difference
 --model_size 6.7 \                  # Target model size
 --quant_type "QAT"                  # Post-Training Quantization(PTQ) or Quantization-Aware Training(QAT)
```

or 

```python
bash ./mixed_bit/run_scripts/QAT/quant_resnet18.sh
```



<font size=6>**QAT**</font>

Because of random seed, bit configuration obtained through feature extraction may have a little difference from ours. Our bit configurations are given in bit_config.py. Our quantized models and logs are also given in [this](https://drive.google.com/drive/folders/1q0wtmWNdqPZuZqnSCQLScYFNIYzXKebg?usp=sharing) link.

```python
#!/usr/bin/env bash
python quant_train.py \
 -a resnet18 \
 --epochs 90 \
 --lr 0.0001 \
 --batch_size 128 \
 --data /Path/to/Dataset/ \
 --save_path /Path/to/Save_quant_model/ \
 --act_range_momentum=0.99 \
 --wd 1e-4 \
 --data_percentage 1 \
 --pretrained \
 --fix_BN \
 --checkpoint_iter -1 \
 --quant_scheme modelsize_6.7_a6_75B
```

or

```python
bash ./QAT/run_scripts/train_resnet18.sh
```



<font size=6>**PTQ**</font>

For the post-training quantization, we only require a few GPU hours to get the quantization model. So we set the random seed. You can directly get the same accuracy in the paper by running codes as follows:

```python
python main_imagenet.py --data_path /Path/to/Dataset/ --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 3, 3, 3, 3]"
```

or

```python
bash ./PTQ/run_scripts/train_resnet18.sh
```



## Related Works

- [BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction (ICLR 2021)](https://arxiv.org/abs/2102.05426)
