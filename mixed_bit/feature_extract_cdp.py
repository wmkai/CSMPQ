import os
import argparse
import numpy as np
import math
from scipy import optimize
import random
import ORM
import torch, time
import torch.nn as nn
from run_manager import RunManager
from models import ResNet_ImageNet, MobileNetV2, TrainRunConfig
from pulp import *
from utils.pytorch_utils import DFS_bit
import pulp
from cdp import calculate_cdp, feature_preprocess
    # get_threshold_by_flops, get_threshold_by_sparsity
    # TFIDFPruner,acculumate_feature
import pdb
import copy
parser = argparse.ArgumentParser()

""" model config """
parser.add_argument('--path', type=str)
parser.add_argument('--model', type=str, default="resnet18",
                    choices=['resnet50', 'mobilenetv2', 'mobilenet', 'resnet18'])
parser.add_argument('--cfg', type=str, default="None")
parser.add_argument('--manual_seed', default=0, type=int)
parser.add_argument("--model_size", default=0, type=float)
parser.add_argument("--beta", default=1, type=float)
parser.add_argument('--quant_type', type=str, default='QAT',
                    choices=['QAT', 'PTQ'])

""" dataset config """
parser.add_argument('--dataset', type=str, default='imagenet',
                    choices=['cifar10', 'imagenet'])
parser.add_argument('--save_path', type=str, default='/home/data/imagenet')

""" runtime config """
parser.add_argument('--gpu', help='gpu available', default='0')
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--n_worker', type=int, default=24)
parser.add_argument("--local_rank", default=0, type=int)

""" cdp config """
parser.add_argument('--coe', type=int, help='whether to use balance coefficient')

if __name__ == '__main__':
    args = parser.parse_args()

    # cpu_num = 1
    # os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    # os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    # os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    # os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    # os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    # torch.set_num_threads(cpu_num)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(1)
    

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    # distributed setting
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()

    # prepare run config
    run_config_path = '%s/run.config' % args.path

    run_config = TrainRunConfig(
        **args.__dict__
    )
    if args.local_rank == 0:
        print('Run config:')
        for k, v in args.__dict__.items():
            print('\t%s: %s' % (k, v))

    if args.model == "resnet18":
        assert args.dataset == 'imagenet', 'resnet18 only supports imagenet dataset'
        net = ResNet_ImageNet(
            depth=18, num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg))
    elif args.model == "resnet50":
        assert args.dataset == 'imagenet', 'resnet50 only supports imagenet dataset'
        net = ResNet_ImageNet(
            depth=50, num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg))
    elif args.model == "mobilenetv2":
        assert args.dataset == 'imagenet', 'mobilenetv2 only supports imagenet dataset'
        net = MobileNetV2(
            num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg))
    

    # build run manager
    run_manager = RunManager(args.path, net, run_config)

    # load checkpoints
    best_model_path = '%s/checkpoint/model_best.pth.tar' % args.path
    assert os.path.isfile(best_model_path), 'wrong path'
    if torch.cuda.is_available():
        checkpoint = torch.load(best_model_path)
    else:
        checkpoint = torch.load(best_model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    run_manager.net.load_state_dict(checkpoint)
    output_dict = {}

    # feature extract
    # start = time.time()
    data_loader = run_manager.run_config.train_loader
    data = next(iter(data_loader))
    data = data[0]
    n = data.size()[0]

    with torch.no_grad():
        feature = net.feature_extract(data, args.quant_type)

    # for i in range(len(feature)):
    #     feature[i] = feature[i].view(n, -1)
    #     feature[i] = feature[i].data.cpu().numpy()
    # pdb.set_trace()
    feature_iit = feature_preprocess(feature)    
    tf_idf_map = calculate_cdp(feature_iit,args.coe)
    # pdb.set_trace()
    # threshold = get_threshold_by_sparsity(tf_idf_map,sparsity)

    # orthogonal_matrix = np.zeros((len(feature), len(feature)))

    # for i in range(len(feature)):
    #     for j in range(len(feature)):
    #         with torch.no_grad():
    #             orthogonal_matrix[i][j] = ORM.orm(ORM.gram_linear(feature[i]), ORM.gram_linear(feature[j]))

    # def sum_list(a, j):
    #     b = 0
    #     for i in range(len(a)):
    #         if i != j:
    #             b += a[i]
    #     return b

    theta = []
    gamma = []
    flops = []

    # for i in range(len(feature)):
    #     gamma.append( sum_list(orthogonal_matrix[i], i) )

    # e^-x
    # for i in range(len(feature)):
    #     theta.append( 1 * math.exp(-1* args.beta *gamma[i]) )
    # theta = np.array(theta)
    # theta = np.negative(theta)
    gamma = np.array(tf_idf_map)

    if args.quant_type == 'QAT':
        theta = copy.deepcopy(gamma)
    elif args.quant_type == 'PTQ':
        # for i in range(len(gamma)):
        #     theta.append( 1 * math.exp(-1* args.beta *gamma[i]) )
        theta = copy.deepcopy(gamma)
        theta = np.array(theta)
    # for x in gamma:
    #     print('%.4f '%x)

    # theta = np.array(tf_idf_map)
    for x in theta:
        print('%.4f '%x)
    theta = np.negative(theta)
    
    # theta = array([0.01322103, 0.01361509, 0.01278478, 0.01267193, 0.00762108, 0.00755147, 0.00914501, 0.00717208, 0.00505306, 0.00479445,
    # 0.00673114, 0.00456354, 0.00451991, 0.00461229, 0.00489852, 0.00488713])

    length = len(feature)
    # layerwise
    params, first_last_size = net.cfg2params_perlayer(net.cfg, length, args.quant_type)
    FLOPs, first_last_flops = net.cfg2flops_layerwise(net.cfg, length, args.quant_type)
    params = [i/(1024*1024) for i in params]
    first_last_size = first_last_size/(1024*1024)


    # Objective function
    # def func(x, sign=1.0):
    #     """ Objective function """
    #     global theta,length
    #     sum_fuc =[]
    #     for i in range(length):
    #         temp = 0.
    #         for j in range(i,length):
    #             temp += theta[j]
    #         sum_fuc.append( x[i] * (sign * temp / (length-i)) )

    #     return sum(sum_fuc)
    # 修改后
    def func(x, sign=1.0):
        """ Objective function """
        global theta,length
        sum_fuc =[]
        for i in range(length):
            sum_fuc.append( x[i] * (sign * theta[i]) )
        return sum(sum_fuc)

    # Derivative function of objective function
    # def func_deriv(x, sign=1.0):
    #     """ Derivative of objective function """
    #     global theta, length
    #     diff = []
    #     for i in range(length):
    #         temp1 = 0.
    #         for j in range(i, length):
    #             temp1 += theta[j]
    #         diff.append(sign * temp1 / (length - i))

    #     return np.array(diff)
    # 修改后
    def func_deriv(x, sign=1.0):
        """ Derivative of objective function """
        global theta, length
        diff = []
        for i in range(length):
            diff.append(sign * theta[i])

        return np.array(diff)
    

    # Constraint function
    def constrain_func(x):
        """ constrain function """
        global params, length
        a = []
        for i in range(length):
            a.append(x[i] * params[i])
        return np.array([args.model_size - first_last_size - sum(a)])

    bnds = [] # bit search space: (0.25,0.5) for PTQ and (0.5,1.0) for QAT , 0.25表示2bit，0.5表示4bit，1.0表示8bit
    if args.quant_type == 'PTQ': #PTQ对所有层搜索空间都是2~4bit
        for i in range(length):
            bnds.append((0.25, 0.5))
    else:
        for i in range(length): #QAT对首尾两层限制为8bit，对其他层搜索空间为4~8bit
            bnds.append((0.5, 1.0))

    bnds = tuple(bnds)
    cons = ({'type': 'ineq',
             'fun': constrain_func}
            )

    result = optimize.minimize(func,x0=[1 for i in range(length)], jac=func_deriv, method='SLSQP', bounds=bnds, constraints=cons)
    # pdb.set_trace()

    if args.model == "resnet18": #resnet18要特殊处理一下
        prun_bitcfg, _ = DFS_bit(result.x[::-1] * 8, [params[length - i - 1] for i in range(length)])
        prun_bitcfg = [prun_bitcfg[length - i - 1] for i in range(length)]
    else:
        prun_bitcfg = np.around(result.x * 8) #其他网络直接使用result.x * 8取整得到bitconfig
    # end = time.time()
    # print("Use", end - start, "seconds. ")


    optimize_cfg = []
    if type(prun_bitcfg[0]) != int:
        for i in range(len(prun_bitcfg)):
            b = list(prun_bitcfg)[i].tolist()
            optimize_cfg.append(int(b))
    else:
        optimize_cfg =prun_bitcfg
    # print(result.x)
    # pdb.set_trace()
    print(optimize_cfg)
    print("Quantization model is", np.sum(np.array(optimize_cfg) * np.array(params) / 8) + first_last_size, "Mb")
    print("Original model is", np.sum(np.array(params)) / 8 * 32 + first_last_size / 8 * 32 , "Mb") 
    print('Quantization model BOPs is',
          (first_last_flops * 8*8 + sum([FLOPs[i] * optimize_cfg[i] *5 for i in range(length)])) / 1e9)








