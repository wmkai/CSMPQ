import os
import sys
import copy
import random
import argparse
import functools
import numpy as np
from datetime import datetime
import pdb
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from nni.algorithms.compression.pytorch.pruning \
    import L1FilterPruner, L1FilterPrunerMasker

# class TFIDFMasker(L1FilterPrunerMasker):
#     def __init__(self, model, pruner, threshold, tf_idf_map, preserve_round=1, dependency_aware=False):
#         super().__init__(model, pruner, preserve_round, dependency_aware)
#         self.threshold=threshold
#         self.tf_idf_map=tf_idf_map
        
#     def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx, channel_masks=None):
#         # get the l1-norm sum for each filter
#         w_tf_idf_structured = self.get_tf_idf_mask(wrapper, wrapper_idx)
        
#         mask_weight = torch.gt(w_tf_idf_structured, self.threshold)[
#             :, None, None, None].expand_as(weight).type_as(weight)
#         mask_bias = torch.gt(w_tf_idf_structured, self.threshold).type_as(
#             weight).detach() if base_mask['bias_mask'] is not None else None

#         return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}
    
#     def get_tf_idf_mask(self, wrapper, wrapper_idx):
#         name = wrapper.name
#         if wrapper.name.split('.')[-1] == 'module':
#             name = wrapper.name[0:-7]
#         #print(name)
#         w_tf_idf_structured = self.tf_idf_map[name]
#         return w_tf_idf_structured


# class TFIDFPruner(L1FilterPruner):
#     def __init__(self, model, config_list, cdp_config:dict, pruning_algorithm='l1', optimizer=None, **algo_kwargs):
#         super().__init__(model, config_list, optimizer)
#         self.set_wrappers_attribute("if_calculated", False)
#         self.masker = TFIDFMasker(model, self, threshold=cdp_config["threshold"], tf_idf_map=cdp_config["map"], **algo_kwargs)
#     def update_masker(self,model,threshold,mapper):
#         self.masker = TFIDFMasker(model, self, threshold=threshold, tf_idf_map=mapper)        

def feature_preprocess(feature):
    for i in range(len(feature)): #([10000, 64, 16, 16])
        # feature[i] = F.relu(feature[i]) #已经在relu后了，所以注释掉
        # pdb.set_trace()
        if len(feature[i].size()) == 4: #卷积层输出
            feature[i] = F.avg_pool2d(feature[i], feature[i].size()[3])  #([10000, 64, 1, 1])
        # fc层输出本来就是二维的，不用变换
        feature[i] = feature[i].view(feature[i].size()[0], -1) #([10000, 64])
        feature[i] = feature[i].transpose(0, 1) #([64, 10000])
    return feature

# def acculumate_feature(model, loader, stop:int):
#     model=model.cuda()
#     features = {}
    
#     def hook_func(m, x, y, name, feature_iit):
#         #print(name, y.shape)
#         f = F.relu(y)
#         #f = y
#         feature = F.avg_pool2d(f, f.size()[3])
#         feature = feature.view(f.size()[0], -1)
#         feature = feature.transpose(0, 1)
#         if name not in feature_iit:
#             feature_iit[name] = feature.cpu()
#         else:
#             feature_iit[name] = torch.cat([feature_iit[name], feature.cpu()], 1)
            
#     hook=functools.partial(hook_func, feature_iit=features)
    
#     handler_list=[]
#     for name, m in model.named_modules():
#         if isinstance(m, nn.Conv2d):
#         #if not isinstance(m, nn.Linear):
#             handler = m.register_forward_hook(functools.partial(hook, name=name))
#             handler_list.append(handler)
#     for batch_idx, (inputs, targets) in enumerate(loader):
#         if batch_idx % (stop//10) == 0:
#             print(batch_idx)
#         if batch_idx >= stop:
#             break
#         model.eval()
#         with torch.no_grad():
#             model(inputs.cuda())
    
#     [ k.remove() for k in handler_list]
#     return features
    
def calc_tf_idf(feature:dict, coe:int, tf_idf_map:dict):   # feature = [c, n] ([64, 10000])    name = 'conv_bn.conv' 
    # calc tf
    # pdb.set_trace()
    balance_coe = np.log((feature.shape[0]/coe)*np.e) if coe else 1.0 #文中公式(8)，float型标量
    # calc idf
    sample_quant = float(feature.shape[1]) #10000
    sample_mean = feature.mean(dim=1).view(feature.shape[0], 1) #得到每个通道的均值 ([64, 1])  
    sample_inverse = (feature >= sample_mean).sum(dim=1).type(torch.FloatTensor) #([64])，文章的Si*，表示每个通道中，比sample_mean大的个数有多少个
    
    # calc tf mean
    feature_sum = feature.sum(dim=0) #([10000])，每个通道累加
    tf = (feature / feature_sum) * balance_coe #文中公式(8) ([64, 10000]) 
    tf_mean = (tf * (feature >= sample_mean)).sum(dim=1)   # Sa ([64])
    tf_mean /= (feature >= sample_mean).sum(dim=1) # ([64])

    idf = torch.log(sample_quant / (sample_inverse + 1.0)) #文中公式(7) ([64])
    idf = idf.cuda() #新增，idf必须和tf_mean都在gpu上
    # pdb.set_trace()
    importance = tf_mean * idf #文中公式(10)  ([64]) 每个输出通道对应一个importance值
    importance = importance.mean().item() #新增
    tf_idf_map.append(importance)

def calculate_cdp(features:dict, coe:int):
    tf_idf_map = []
    for feature in features:
        calc_tf_idf(feature, coe=coe, tf_idf_map=tf_idf_map)
    return tf_idf_map


