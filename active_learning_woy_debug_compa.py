import torch
import torch.nn as nn
import torch.optim as optim
from utils.train_utils import get_pred_logit
import torchvision.transforms as transforms
import npy_dataset
import argparse
import numpy as np
from models.resnet import *
import os
import random
import pdb
import matplotlib.pyplot as plt
from models.simple_cnn import Vanillann
from models.densenet import DenseNet121
from models.mobilenet import MobileNetV2
from sklearn.cluster import KMeans
from copy import deepcopy
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--model', type=str, default='resnet20', help="name of save log") 
parser.add_argument('--optimizer', type=str, default='adamw', help="adamw/sgd") 
parser.add_argument('--pred_path', type=str, default='pretrain_resnet50_1e-8_256_0', help="adamw/sgd") 
parser.add_argument('--dataset', type=str, default='cifar', help="cifar/svhn") 
parser.add_argument('--test_mode', type=str, default='oracle_sampling_cut_whole', help="oracle_sampling/random/predict/full") 
parser.add_argument('--epochs', type=int, default=150) 
parser.add_argument('--T_max', type=int, default=50) 
parser.add_argument('--eta_min', default=0, type=float, help='min lr')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay')
parser.add_argument('--seed', type=int, default=1) 
parser.add_argument('--train_batch_size', type=int, default=128) 
parser.add_argument('--weight_decay', type=float, default=5e-4) 
parser.add_argument('--sample_number', type=int, default=300) 
parser.add_argument('--schedule_type', type=str, default="cosine", help="step/cosine/poly") 
parser.add_argument('--loss_type', type=str, default="base", help="base/reweight/reweight_clip") 
parser.add_argument('--weight_folder', type=str, default='active_weight_binary/') 
parser.add_argument('--weight_name', type=str, default='active_weight.pth') 
parser.add_argument('--constant_1', type=float, default=100) 
parser.add_argument('--constant_2', type=float, default=0) 
parser.add_argument('--constant_3', type=float, default=0.5) 
parser.add_argument('--cut_thre', type=float, default=5) 
parser.add_argument('--batch_size', type=int, default=256) 
parser.add_argument('--un_folder', type=str, default='cifar_npy') 
args = parser.parse_args()
pic_path = "./pics_" + args.dataset
os.makedirs(pic_path, exist_ok=True)
args.pic_name = os.path.join(pic_path, args.weight_name.replace(".pth", ".png"))
pred_base_path = "binary_prediction/"


def cal_un_woy(logit):
    covariance = np.matmul((logit-np.mean(logit, axis=0, keepdims=True)).transpose()[1:, :], (logit-np.mean(logit, axis=0, keepdims=True))[:, 1:])/(len(logit)-1)
    
    logit =logit -np.max(logit, 1, keepdims=True)
    pred = np.mean(np.exp(logit)/np.sum(np.exp(logit), 1, keepdims=True), 0)[1:]

    pred_ = np.reshape(pred, (len(pred), 1))
    psi = np.diag(pred) - np.matmul(pred_, pred_.transpose())
    met = np.trace(np.matmul(psi, covariance))
    return met
# def cal_un_woy(logit):
#     covariance = np.matmul((logit-np.mean(logit, axis=0, keepdims=True)).transpose(), (logit-np.mean(logit, axis=0, keepdims=True)))/(len(logit)-1)
#     # logit = logit
#     logit =logit -np.max(logit, 1, keepdims=True)
#     pred =np.mean(np.exp(logit)/np.sum(np.exp(logit), 1),0)
#     pred_ = np.reshape(pred, (len(pred), 1))
#     psi = np.diag(pred) - np.matmul(pred_, pred_.transpose())
#     met = np.trace(np.matmul(psi, covariance))
#     return met
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(args.seed)
os.makedirs(args.weight_folder, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.dataset == "svhn":
    class_n=10
    mean = [x / 255 for x in [109.9, 109.7, 113.8]]
    std = [x / 255 for x in [50.1, 50.6, 50.8]]
    transform_list = [transforms.RandomCrop(32, padding=2, fill=128)]
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std))
    transform_train = transforms.Compose(transform_list)
    transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
    train_data =np.load( "svhn_npy/train_X_2.npy")
    train_lab = np.load("svhn_npy/train_Y_2.npy")
    test_data =np.load( "svhn_npy/train_X_1.npy")
    test_lab = np.load("svhn_npy/train_Y_1.npy") 
    real_test_data =np.load( "svhn_npy/test_X.npy")
    real_test_lab = np.load("svhn_npy/test_Y.npy")
    # train_lab_oh = np.zeros((len(train_lab), 10))
    # for k in range(len(train_lab_oh)):
    #     train_lab_oh[k, train_lab[k]] = 1
    train_logit = np.load(os.path.join(args.un_folder, "train_UN_2_raw.npy"))
    train_logit = train_logit.transpose((1,0,2))
    train_metric = np.zeros((len(train_logit), ))
    for k in range(len(train_logit)):
        train_metric[k] = cal_un_woy(train_logit[k], )
    print(len(train_data), len(test_data), len(real_test_data))
elif args.dataset == "cifar100":
    class_n = 100
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    train_data =np.load("cifar100_npy/train_X_2.npy")
    train_lab = np.load("cifar100_npy/train_Y_2.npy")
    test_data =np.load( "cifar100_npy/train_X_1.npy")
    test_lab = np.load("cifar100_npy/train_Y_1.npy")
    real_test_data =np.load( "cifar100_npy/test_X.npy")
    real_test_lab = np.load("cifar100_npy/test_Y.npy")
    train_lab_oh = np.zeros((len(train_lab), 100))
    for k in range(len(train_lab_oh)):
        train_lab_oh[k, train_lab[k]] = 1
    train_logit = np.load(os.path.join("cifar100_npy_res20", "train_UN_2_raw.npy"))
    # train_logit = train_logit.transpose((1,0,2))
    # train_metric = np.zeros((len(train_logit), ))
    # for k in range(len(train_logit)):
    #     train_metric[k] = cal_un_woy(train_logit[k] )
        
    # train_logit = np.load(os.path.join(args.un_folder, "train_UN_2_raw.npy"))
    # train_logit = train_logit.transpose((1,0,2))
    train_pseudo_label = np.argmax(np.mean(train_logit, 0), 1)
    print(train_pseudo_label)
    for j in range(100):
        print(len([i for i in range(len(train_pseudo_label)) if train_pseudo_label[i]==j]))
    # sss
    if  args.test_mode != 'iwes' and args.test_mode != 'entropy' and args.test_mode != 'coreset' and  args.test_mode != 'badge' and args.test_mode != 'least_confidence':
        train_logit = train_logit.transpose((1,0,2))
        train_metric = np.zeros((len(train_logit), ))
        for k in range(len(train_logit)):
            train_metric[k] = cal_un_woy(train_logit[k] )
    elif args.test_mode == 'iwes':
        train_logit = train_logit[0]
        print(train_logit.shape)
        train_logit =train_logit -np.max(train_logit, 1, keepdims=True)
        train_logit = np.exp(train_logit)/np.sum(np.exp(train_logit), 1 ,keepdims=True)
        train_metric = - train_logit * np.log2(train_logit) /np.log2(10)
        train_metric = np.sum(train_metric, axis=1)
    elif args.test_mode == 'entropy':
        train_logit = train_logit[0]
        print(train_logit.shape)
        train_logit =train_logit -np.max(train_logit, 1, keepdims=True)
        train_logit = np.exp(train_logit)/np.sum(np.exp(train_logit), 1 ,keepdims=True)
        train_metric = - train_logit * np.log2(train_logit) 
        train_metric = np.sum(train_metric, axis=1)
        print(train_metric.shape)
        print(train_metric)
    elif args.test_mode == 'coreset':
        train_logit = np.load(os.path.join("cifar100_npy_res20", "train_UN_2_feature.npy"))
        train_logit = train_logit[0]
    elif args.test_mode == 'least_confidence' :
        train_lab_oh = np.zeros((len(train_lab), 100))
        for k in range(len(train_lab_oh)):
            train_lab_oh[k, train_lab[k]] = 1
        train_logit = np.load(os.path.join("cifar100_npy_res20", "train_UN_2_raw.npy"))
        train_logit1 = train_logit[0]
        train_logit1 = train_logit1 -np.max(train_logit1, 1, keepdims=True)
        train_logit1 = np.exp(train_logit1)/np.sum(np.exp(train_logit1), 1 ,keepdims=True)
        train_metric = 1-np.max(train_logit1, 1)
        # print(train_logit.shape)
        # ssss
        # train_feature = train_feature[0]
        # train_logit = np.load(os.path.join(args.un_folder, "train_embed_2.npy"))
    elif args.test_mode == 'badge':
        train_logit = np.load(os.path.join("cifar100_npy_res20", "train_UN_2_raw.npy"))
        train_feature = np.load(os.path.join("cifar100_npy_res20", "train_UN_2_feature.npy"))
        train_logit = train_logit[0]
        train_feature = train_feature[0]
        train_logit = train_logit -np.max(train_logit, 1, keepdims=True)
        train_logit = np.exp(train_logit)/np.sum(np.exp(train_logit), 1 ,keepdims=True)
        # maxInds = train_lab
        maxInds = np.argmax(train_logit,1)
        train_metric = np.zeros((len(train_logit), 64* 100))
        embDim= 64
        for j in range(len(train_logit)):
            for c in range(100):
                if c == maxInds[j]:
                    train_metric[j][embDim * c : embDim * (c+1)] = deepcopy(train_feature[j]) * (1 - train_logit[j][c])
                else:
                     train_metric[j][embDim * c : embDim * (c+1)] = deepcopy(train_feature[j]) * (-1 * train_logit[j][c])

elif args.dataset == "cifar100n":
    class_n = 100
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    train_data =np.load("cifar100_npy/train_X_2.npy")
    train_lab = np.load("cifar100_npy/train_noise_Y_2.npy")
    test_data =np.load( "cifar100_npy/train_X_1.npy")
    test_lab = np.load("cifar100_npy/train_noise_Y_1.npy")
    real_test_data =np.load( "cifar100_npy/test_X.npy")
    real_test_lab = np.load("cifar100_npy/test_Y.npy")
    train_lab_oh = np.zeros((len(train_lab), 100))
    for k in range(len(train_lab_oh)):
        train_lab_oh[k, train_lab[k]] = 1
    train_logit = np.load(os.path.join("cifar100n_npy_res20", "train_UN_2_raw.npy"))
    # train_logit = train_logit.transpose((1,0,2))
    # train_metric = np.zeros((len(train_logit), ))
    # for k in range(len(train_logit)):
    #     train_metric[k] = cal_un_woy(train_logit[k] )
        
    # train_logit = np.load(os.path.join(args.un_folder, "train_UN_2_raw.npy"))
    # train_logit = train_logit.transpose((1,0,2))
    train_pseudo_label = np.argmax(np.mean(train_logit, 0), 1)
    print(train_pseudo_label)
    for j in range(100):
        print(len([i for i in range(len(train_pseudo_label)) if train_pseudo_label[i]==j]))
    # sss
    if  args.test_mode != 'iwes' and args.test_mode != 'entropy' and args.test_mode != 'coreset' and  args.test_mode != 'badge' and args.test_mode != 'least_confidence':
        train_logit = train_logit.transpose((1,0,2))
        train_metric = np.zeros((len(train_logit), ))
        for k in range(len(train_logit)):
            train_metric[k] = cal_un_woy(train_logit[k] )
    elif args.test_mode == 'iwes':
        train_logit = train_logit[0]
        print(train_logit.shape)
        train_logit =train_logit -np.max(train_logit, 1, keepdims=True)
        train_logit = np.exp(train_logit)/np.sum(np.exp(train_logit), 1 ,keepdims=True)
        train_metric = - train_logit * np.log2(train_logit) /np.log2(10)
        train_metric = np.sum(train_metric, axis=1)
    elif args.test_mode == 'entropy':
        train_logit = train_logit[0]
        print(train_logit.shape)
        train_logit =train_logit -np.max(train_logit, 1, keepdims=True)
        train_logit = np.exp(train_logit)/np.sum(np.exp(train_logit), 1 ,keepdims=True)
        train_metric = - train_logit * np.log2(train_logit) 
        train_metric = np.sum(train_metric, axis=1)
        print(train_metric.shape)
        print(train_metric)
    elif args.test_mode == 'coreset':
        train_logit = np.load(os.path.join("cifar100n_npy_res20", "train_UN_2_feature.npy"))
        train_logit = train_logit[0]
    elif args.test_mode == 'least_confidence' :
        train_lab_oh = np.zeros((len(train_lab), 100))
        for k in range(len(train_lab_oh)):
            train_lab_oh[k, train_lab[k]] = 1
        train_logit = np.load(os.path.join("cifar100n_npy_res20", "train_UN_2_raw.npy"))
        train_logit1 = train_logit[0]
        train_logit1 = train_logit1 -np.max(train_logit1, 1, keepdims=True)
        train_logit1 = np.exp(train_logit1)/np.sum(np.exp(train_logit1), 1 ,keepdims=True)
        train_metric = 1-np.max(train_logit1, 1)
        # print(train_metric.shape)
        # print(train_logit.shape)
        # ssss
        # train_feature = train_feature[0]
        # train_logit = np.load(os.path.join(args.un_folder, "train_embed_2.npy"))
    elif args.test_mode == 'badge':
        train_logit = np.load(os.path.join("cifar100n_npy_res20", "train_UN_2_raw.npy"))
        train_feature = np.load(os.path.join("cifar100n_npy_res20", "train_UN_2_feature.npy"))
        train_logit = train_logit[0]
        train_feature = train_feature[0]
        train_logit = train_logit -np.max(train_logit, 1, keepdims=True)
        train_logit = np.exp(train_logit)/np.sum(np.exp(train_logit), 1 ,keepdims=True)
        # maxInds = train_lab
        maxInds = np.argmax(train_logit,1)
        train_metric = np.zeros((len(train_logit), 64* 100))
        embDim= 64
        for j in range(len(train_logit)):
            for c in range(100):
                if c == maxInds[j]:
                    train_metric[j][embDim * c : embDim * (c+1)] = deepcopy(train_feature[j]) * (1 - train_logit[j][c])
                else:
                     train_metric[j][embDim * c : embDim * (c+1)] = deepcopy(train_feature[j]) * (-1 * train_logit[j][c])
    



elif args.dataset == "cifar":
    class_n = 10
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    train_data =np.load("cifar_npy/train_X_2.npy")
    train_lab = np.load("cifar_npy/train_Y_2.npy")
    test_data =np.load( "cifar_npy/train_X_1.npy")
    test_lab = np.load("cifar_npy/train_Y_1.npy")
    real_test_data =np.load( "cifar_npy/test_X.npy")
    real_test_lab = np.load("cifar_npy/test_Y.npy")
    # train_lab_oh = np.zeros((len(train_lab), 10))
    # for k in range(len(train_lab_oh)):
    #     train_lab_oh[k, train_lab[k]] = 1
    train_logit = np.load(os.path.join(args.un_folder, "train_UN_2_raw.npy"))
    train_pseudo_label = np.argmax(np.mean(train_logit, 0), 1)
    print(train_pseudo_label)
    for j in range(10):
        print(len([i for i in range(len(train_pseudo_label)) if train_pseudo_label[i]==j]))
    # sss
    if  args.test_mode != 'iwes' and args.test_mode != 'entropy' and args.test_mode != 'coreset' and  args.test_mode != 'badge' and args.test_mode != 'least_confidence':
        train_logit = train_logit.transpose((1,0,2))
        train_metric = np.zeros((len(train_logit), ))
        for k in range(len(train_logit)):
            train_metric[k] = cal_un_woy(train_logit[k] )
    elif args.test_mode == 'iwes':
        train_logit = train_logit[0]
        print(train_logit.shape)
        train_logit =train_logit -np.max(train_logit, 1, keepdims=True)
        train_logit = np.exp(train_logit)/np.sum(np.exp(train_logit), 1 ,keepdims=True)
        train_metric = - train_logit * np.log2(train_logit) /np.log2(10)
        train_metric = np.sum(train_metric, axis=1)
    elif args.test_mode == 'entropy':
        train_logit = train_logit[0]
        print(train_logit.shape)
        train_logit =train_logit -np.max(train_logit, 1, keepdims=True)
        train_logit = np.exp(train_logit)/np.sum(np.exp(train_logit), 1 ,keepdims=True)
        train_metric = - train_logit * np.log2(train_logit) 
        train_metric = np.sum(train_metric, axis=1)
        print(train_metric.shape)
        print(train_metric)
    elif args.test_mode == 'coreset':
        train_logit = np.load(os.path.join(args.un_folder, "train_UN_2_feature.npy"))
        train_logit = train_logit[0]
        # print(train_logit.shape)
        # ssss
        # train_feature = train_feature[0]
        # train_logit = np.load(os.path.join(args.un_folder, "train_embed_2.npy"))
    elif args.test_mode == 'badge':
        train_logit = np.load(os.path.join(args.un_folder, "train_UN_2_raw.npy"))
        train_feature = np.load(os.path.join(args.un_folder, "train_UN_2_feature.npy"))
        train_logit = train_logit[0]
        train_feature = train_feature[0]
        train_logit = train_logit -np.max(train_logit, 1, keepdims=True)
        train_logit = np.exp(train_logit)/np.sum(np.exp(train_logit), 1 ,keepdims=True)
        # maxInds = train_lab
        maxInds = np.argmax(train_logit,1)
        train_metric = np.zeros((len(train_logit), 64* 10))
        embDim= 64
        for j in range(len(train_logit)):
            for c in range(10):
                if c == maxInds[j]:
                    train_metric[j][embDim * c : embDim * (c+1)] = deepcopy(train_feature[j]) * (1 - train_logit[j][c])
                else:
                     train_metric[j][embDim * c : embDim * (c+1)] = deepcopy(train_feature[j]) * (-1 * train_logit[j][c])
        
        # train_logit = train_lab_oh * train_logit
    elif args.test_mode == 'least_confidence' :
        train_lab_oh = np.zeros((len(train_lab), 10))
        for k in range(len(train_lab_oh)):
            train_lab_oh[k, train_lab[k]] = 1
        train_logit = np.load(os.path.join(args.un_folder, "train_UN_2_raw.npy"))
        train_logit1 = train_logit[0]
        train_logit1 = train_logit1 -np.max(train_logit1, 1, keepdims=True)
        train_logit1 = np.exp(train_logit1)/np.sum(np.exp(train_logit1), 1 ,keepdims=True)
        train_metric = 1-np.max(train_logit1, 1)
        # print(train_metric.shape)
        # ssss


elif args.dataset == "cifar10_worse":
    class_n = 10
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    train_data =np.load("cifar_noise_npy/train_X_2.npy")
    train_lab = np.load("cifar_noise_npy/train_Y_2_wrost.npy")
    test_data =np.load( "cifar_noise_npy/train_X_1.npy")
    test_lab = np.load("cifar_noise_npy/train_Y_1_wrost.npy")
    real_test_data =np.load( "cifar_noise_npy/test_X.npy")
    real_test_lab = np.load("cifar_noise_npy/test_Y.npy")
    
    train_logit = np.load(os.path.join("cifar10_worse_npy", "train_UN_2_raw.npy"))
    train_pseudo_label = np.argmax(np.mean(train_logit, 0), 1)
    train_logit = train_logit.transpose((1,0,2))
    train_metric = np.zeros((len(train_logit), ))
    
    if  args.test_mode != 'iwes' and args.test_mode != 'entropy' and args.test_mode != 'coreset' and  args.test_mode != 'badge' and args.test_mode != 'least_confidence':
 #       train_logit = train_logit.transpose((1,0,2))
        train_metric = np.zeros((len(train_logit), ))
        for k in range(len(train_logit)):
            train_metric[k] = cal_un_woy(train_logit[k] )
    elif args.test_mode == 'iwes':
        train_logit = train_logit[0]
        print(train_logit.shape)
        train_logit =train_logit -np.max(train_logit, 1, keepdims=True)
        train_logit = np.exp(train_logit)/np.sum(np.exp(train_logit), 1 ,keepdims=True)
        train_metric = - train_logit * np.log2(train_logit) /np.log2(10)
        train_metric = np.sum(train_metric, axis=1)
    elif args.test_mode == 'entropy':
        train_logit = train_logit[0]
        print(train_logit.shape)
        train_logit =train_logit -np.max(train_logit, 1, keepdims=True)
        train_logit = np.exp(train_logit)/np.sum(np.exp(train_logit), 1 ,keepdims=True)
        train_metric = - train_logit * np.log2(train_logit) 
        train_metric = np.sum(train_metric, axis=1)
        print(train_metric.shape)
        print(train_metric)
    elif args.test_mode == 'coreset':
        train_logit = np.load(os.path.join("cifar10_worse_npy", "train_UN_2_feature.npy"))
        train_logit = train_logit[0]
        # print(train_logit.shape)
        # ssss
        # train_feature = train_feature[0]
        # train_logit = np.load(os.path.join(args.un_folder, "train_embed_2.npy"))
    elif args.test_mode == 'badge':
        train_logit = np.load(os.path.join("cifar10_worse_npy", "train_UN_2_raw.npy"))
        train_feature = np.load(os.path.join("cifar10_worse_npy", "train_UN_2_feature.npy"))
        train_logit = train_logit[0]
        train_feature = train_feature[0]
        train_logit = train_logit -np.max(train_logit, 1, keepdims=True)
        train_logit = np.exp(train_logit)/np.sum(np.exp(train_logit), 1 ,keepdims=True)
        # maxInds = train_lab
        maxInds = np.argmax(train_logit,1)
        train_metric = np.zeros((len(train_logit), 64* 10))
        embDim= 64
        for j in range(len(train_logit)):
            for c in range(10):
                if c == maxInds[j]:
                    train_metric[j][embDim * c : embDim * (c+1)] = deepcopy(train_feature[j]) * (1 - train_logit[j][c])
                else:
                     train_metric[j][embDim * c : embDim * (c+1)] = deepcopy(train_feature[j]) * (-1 * train_logit[j][c])
        
        # train_logit = train_lab_oh * train_logit
    elif args.test_mode == 'least_confidence' :
        train_lab_oh = np.zeros((len(train_lab), 10))
        for k in range(len(train_lab_oh)):
            train_lab_oh[k, train_lab[k]] = 1
        train_logit = np.load(os.path.join("cifar10_worse_npy", "train_UN_2_raw.npy"))
        train_logit1 = train_logit[0]
        train_logit1 = train_logit1 -np.max(train_logit1, 1, keepdims=True)
        train_logit1 = np.exp(train_logit1)/np.sum(np.exp(train_logit1), 1 ,keepdims=True)
        train_metric = 1-np.max(train_logit1, 1)
        # print(train_metric.shape)
        # ssss

    # for k in range(len(train_logit)):
    #     train_metric[k] = cal_un_woy(train_logit[k] )
        

    print(len(train_data), len(test_data), len(real_test_data))

elif args.dataset == "cifar10_aggre":
    class_n = 10
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    train_data =np.load("cifar_noise_npy/train_X_2.npy")
    train_lab = np.load("cifar_noise_npy/train_Y_2_aggre.npy")
    test_data =np.load( "cifar_noise_npy/train_X_1.npy")
    test_lab = np.load("cifar_noise_npy/train_Y_1_aggre.npy")
    real_test_data = np.load( "cifar_noise_npy/test_X.npy")
    real_test_lab = np.load("cifar_noise_npy/test_Y.npy")
    train_logit = np.load(os.path.join("cifar10_aggre_npy", "train_UN_2_raw.npy"))
    train_pseudo_label = np.argmax(np.mean(train_logit, 0), 1)
    train_logit = train_logit.transpose((1,0,2))
    if  args.test_mode != 'iwes' and args.test_mode != 'entropy' and args.test_mode != 'coreset' and  args.test_mode != 'badge' and args.test_mode != 'least_confidence':
        # train_logit = train_logit.transpose((1,0,2))
        train_metric = np.zeros((len(train_logit), ))
        for k in range(len(train_logit)):
            train_metric[k] = cal_un_woy(train_logit[k] )
    elif args.test_mode == 'iwes':
        train_logit = train_logit[0]
        print(train_logit.shape)
        train_logit =train_logit -np.max(train_logit, 1, keepdims=True)
        train_logit = np.exp(train_logit)/np.sum(np.exp(train_logit), 1 ,keepdims=True)
        train_metric = - train_logit * np.log2(train_logit) /np.log2(10)
        train_metric = np.sum(train_metric, axis=1)
    elif args.test_mode == 'entropy':
        train_logit = train_logit[0]
        print(train_logit.shape)
        train_logit =train_logit -np.max(train_logit, 1, keepdims=True)
        train_logit = np.exp(train_logit)/np.sum(np.exp(train_logit), 1 ,keepdims=True)
        train_metric = - train_logit * np.log2(train_logit) 
        train_metric = np.sum(train_metric, axis=1)
        print(train_metric.shape)
        print(train_metric)
    elif args.test_mode == 'coreset':
        train_logit = np.load(os.path.join("cifar10_aggre_npy", "train_UN_2_feature.npy"))
        train_logit = train_logit[0]
        # print(train_logit.shape)
        # ssss
        # train_feature = train_feature[0]
        # train_logit = np.load(os.path.join(args.un_folder, "train_embed_2.npy"))
    elif args.test_mode == 'badge':
        train_logit = np.load(os.path.join("cifar10_aggre_npy", "train_UN_2_raw.npy"))
        train_feature = np.load(os.path.join("cifar10_aggre_npy", "train_UN_2_feature.npy"))
        train_logit = train_logit[0]
        train_feature = train_feature[0]
        train_logit = train_logit -np.max(train_logit, 1, keepdims=True)
        train_logit = np.exp(train_logit)/np.sum(np.exp(train_logit), 1 ,keepdims=True)
        # maxInds = train_lab
        maxInds = np.argmax(train_logit,1)
        train_metric = np.zeros((len(train_logit), 64* 10))
        embDim= 64
        for j in range(len(train_logit)):
            for c in range(10):
                if c == maxInds[j]:
                    train_metric[j][embDim * c : embDim * (c+1)] = deepcopy(train_feature[j]) * (1 - train_logit[j][c])
                else:
                     train_metric[j][embDim * c : embDim * (c+1)] = deepcopy(train_feature[j]) * (-1 * train_logit[j][c])
        
        # train_logit = train_lab_oh * train_logit
    elif args.test_mode == 'least_confidence' :
        train_lab_oh = np.zeros((len(train_lab), 10))
        for k in range(len(train_lab_oh)):
            train_lab_oh[k, train_lab[k]] = 1
        train_logit = np.load(os.path.join("cifar10_aggre_npy", "train_UN_2_raw.npy"))
        train_logit1 = train_logit[0]
        train_logit1 = train_logit1 -np.max(train_logit1, 1, keepdims=True)
        train_logit1 = np.exp(train_logit1)/np.sum(np.exp(train_logit1), 1 ,keepdims=True)
        train_metric = 1-np.max(train_logit1, 1)
        # print(train_metric.shape)
        # ssss

    # for k in range(len(train_logit)):
    #     train_metric[k] = cal_un_woy(train_logit[k] )
        
    # train_metric = np.zeros((len(train_logit), ))
    # for k in range(len(train_logit)):
    #     train_metric[k] = cal_un_woy(train_logit[k] )
    # print(len(train_data), len(test_data), len(real_test_data))


elif args.dataset == "cifar10_rand1":
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    train_data =np.load("cifar_noise_npy/train_X_2.npy")
    train_lab = np.load("cifar_noise_npy/train_Y_2_rand1.npy")
    test_data =np.load( "cifar_noise_npy/train_X_1.npy")
    test_lab = np.load("cifar_noise_npy/train_Y_1_rand1.npy")
    real_test_data = np.load( "cifar_noise_npy/test_X.npy")
    real_test_lab = np.load("cifar_noise_npy/test_Y.npy")
    train_logit = np.load(os.path.join("cifar10_rand1_npy", "train_UN_2_raw.npy"))
    train_logit = train_logit.transpose((1,0,2))
    train_metric = np.zeros((len(train_logit), ))
    for k in range(len(train_logit)):
        train_metric[k] = cal_un_woy(train_logit[k] )
    print(len(train_data), len(test_data), len(real_test_data))
elif args.dataset == "cifar10_rand2":
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    train_data =np.load("cifar_noise_npy/train_X_2.npy")
    train_lab = np.load("cifar_noise_npy/train_Y_2_rand2.npy")
    test_data =np.load( "cifar_noise_npy/train_X_1.npy")
    test_lab = np.load("cifar_noise_npy/train_Y_1_rand2.npy")
    real_test_data = np.load( "cifar_noise_npy/test_X.npy")
    real_test_lab = np.load("cifar_noise_npy/test_Y.npy")
    train_logit = np.load(os.path.join("cifar10_rand2_npy", "train_UN_2_raw.npy"))
    train_logit = train_logit.transpose((1,0,2))
    train_metric = np.zeros((len(train_logit), ))
    for k in range(len(train_logit)):
        train_metric[k] = cal_un_woy(train_logit[k] )
    print(len(train_data), len(test_data), len(real_test_data))
elif args.dataset == "cifar10_rand3":
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    train_data =np.load("cifar_noise_npy/train_X_2.npy")
    train_lab = np.load("cifar_noise_npy/train_Y_2_rand3.npy")
    test_data =np.load( "cifar_noise_npy/train_X_1.npy")
    test_lab = np.load("cifar_noise_npy/train_Y_1_rand3.npy")
    real_test_data = np.load( "cifar_noise_npy/test_X.npy")
    real_test_lab = np.load("cifar_noise_npy/test_Y.npy")
    train_logit = np.load(os.path.join("cifar10_rand3_npy", "train_UN_2_raw.npy"))
    train_logit = train_logit.transpose((1,0,2))
    train_metric = np.zeros((len(train_logit), ))
    if  args.test_mode != 'iwes' :
        for k in range(len(train_logit)):
            train_metric[k] = cal_un_woy(train_logit[k] )
    else:
        train_metric = 0
    print(len(train_data), len(test_data), len(real_test_data))
######################### sample_selection
if args.test_mode != 'coreset' and args.test_mode != 'badge':
    train_metric[train_metric<=0] = np.min(train_metric[train_metric>0])
    train_metric = np.sqrt(train_metric)



######################### sample_selection

if args.test_mode == 'random':
    train_ind = []
    train_loss_weight = []
    for k in range(class_n):
        ind =  [i for i in range(len(train_metric)) if train_lab[i] == k]
        tmp_ind = np.random.choice(ind, size=(args.sample_number, ),replace=False)
        train_loss_weight.append(train_metric[tmp_ind])
        train_ind.extend(tmp_ind)
if args.test_mode == 'random_pred':
    train_ind = []
    train_loss_weight = []
    for k in range(class_n):
        ind =  [i for i in range(len(train_metric)) if train_pseudo_label[i] == k]
        tmp_ind = np.random.choice(ind, size=(args.sample_number, ),replace=False)
        train_loss_weight.append(train_metric[tmp_ind])
        train_ind.extend(tmp_ind)
if args.test_mode == 'random_whole':
    train_ind = []
    train_loss_weight = []
    # for k in range(10):
    ind =  [i for i in range(len(train_metric))]
    tmp_ind = np.random.choice(ind, size=(args.sample_number*class_n, ),replace=False)
    train_loss_weight.append(train_metric[tmp_ind])
    train_ind.extend(tmp_ind)
elif args.test_mode == 'oracle_sampling_cut':
    train_metric_ = np.clip(train_metric,a_min=0, a_max=args.constant_1)
    train_ind = []
    train_loss_weight = []
    for k in range(class_n):
        ind =  [i for i in range(len(train_metric)) if train_lab[i] == k]
        tmp_ind = np.random.choice(ind, size=(min(args.sample_number, len(ind)), ),replace=False, p=(train_metric_[ind])/np.sum((train_metric_[ind])))
        train_loss_weight.append(train_metric[tmp_ind])
        train_ind.extend(tmp_ind)
elif args.test_mode == 'oracle_sampling_cut_pred':
    train_metric_ = np.clip(train_metric,a_min=0, a_max=args.constant_1)
    train_ind = []
    train_loss_weight = []
    for k in range(class_n):
        ind =  [i for i in range(len(train_metric)) if train_pseudo_label[i] == k]
        tmp_ind = np.random.choice(ind, size=(min(len(ind),args.sample_number), ),replace=False, p=(train_metric_[ind])/np.sum((train_metric_[ind])))
        train_loss_weight.append(train_metric[tmp_ind])
        train_ind.extend(tmp_ind)
elif args.test_mode == 'oracle_sampling_cut_whole':
    train_metric_ = np.clip(train_metric,a_min=0, a_max=args.constant_1)
    train_ind = []
    train_loss_weight = []
    
    ind =  [i for i in range(len(train_metric)) ]
    print(args.sample_number*class_n)
    print(len(ind))
    tmp_ind = np.random.choice(ind, size=(args.sample_number*class_n, ),replace=False, p=(train_metric_[ind])/np.sum((train_metric_[ind])))
    train_loss_weight.append(train_metric[tmp_ind])
    train_ind.extend(tmp_ind)
    
elif args.test_mode == 'oracle_sampling_whole':
  
    train_ind = []
    train_loss_weight = []
    
    ind =  [i for i in range(len(train_metric)) ]
    print(args.sample_number*class_n)
    print(len(ind))
    tmp_ind = np.random.choice(ind, size=(args.sample_number*class_n, ),replace=False, p=(train_metric[ind]+args.constant_1)/np.sum((train_metric[ind]+args.constant_1)))
    train_loss_weight.append(train_metric[tmp_ind])
    train_ind.extend(tmp_ind)

#entropy
elif args.test_mode == 'entropy':
    train_ind = []
    train_loss_weight = []
    for k in range(class_n):
        ind =  [i for i in range(len(train_metric)) if train_pseudo_label[i] == k]
        tmp_ind = np.random.choice(ind, size=(args.sample_number, ),replace=False, p=(train_metric[ind])/np.sum((train_metric[ind])))
        train_loss_weight.append(train_metric[tmp_ind])
        train_ind.extend(tmp_ind)
elif args.test_mode == 'iwes':
    train_ind = []
    train_loss_weight = []
    for k in range(class_n):
        ind =  [i for i in range(len(train_metric)) if train_pseudo_label[i] == k]
        if len(ind) <= args.sample_number:
            tmp_ind = ind
        else:
            tmp_ind = np.random.choice(ind, size=(args.sample_number, ),replace=False, p=(train_metric[ind])/np.sum((train_metric[ind])))
        train_loss_weight.append(train_metric[tmp_ind])
        train_ind.extend(tmp_ind)
elif args.test_mode == 'least_confidence':
    train_ind = []
    train_loss_weight = []
    for k in range(class_n):
        
        ind =  [i for i in range(len(train_metric)) if train_pseudo_label[i] == k]
        if len(ind) <= args.sample_number:
            tmp_ind = np.random.choice(ind, size=(len(ind), ),replace=False, p=(train_metric[ind])/np.sum((train_metric[ind])))
        else:
            tmp_ind = np.random.choice(ind, size=(args.sample_number, ),replace=False, p=(train_metric[ind])/np.sum((train_metric[ind])))
        train_loss_weight.append(train_metric[tmp_ind])
        train_ind.extend(tmp_ind)

elif args.test_mode == 'coreset' :
    train_ind = []
    train_loss_weight = []
    cluster_learner = KMeans(n_clusters=class_n*args.sample_number, max_iter=30)
    cluster_learner.fit(train_logit)
    cluster_idxs = cluster_learner.predict(train_logit)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (train_logit - centers)**2
    # print(dis)
    # print(train_logit.shape)
    dis = dis.sum(axis=1)
    train_ind = np.array([np.arange(train_logit.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(class_n*args.sample_number)])
    print(len(train_ind))
    # ssss


elif  args.test_mode == 'badge' :
    train_ind = []
    train_loss_weight = []
    cluster_learner = KMeans(n_clusters=class_n*args.sample_number, max_iter=30)
    cluster_learner.fit(train_logit)
    cluster_idxs = cluster_learner.predict(train_logit)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (train_logit - centers)**2
    # print(dis)
    # print(train_logit.shape)
    dis = dis.sum(axis=1)
    train_ind = np.array([np.arange(train_logit.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(class_n*args.sample_number)])
    print(train_ind)
    print(len(train_ind))
    # ssss
elif args.test_mode == 'oracle_sampling':
    train_ind = []
    train_loss_weight = []
    for k in range(class_n):
        ind =  [i for i in range(len(train_metric)) if train_lab[i] == k]
        tmp_ind = np.random.choice(ind, size=(args.sample_number, ),replace=False, p=(train_metric[ind]+args.constant_1)/np.sum((train_metric[ind]+args.constant_1)))
        train_loss_weight.append(train_metric[tmp_ind])
        train_ind.extend(tmp_ind)

elif args.test_mode == 'random_replacement':
    train_ind = []
    train_loss_weight = []
    for k in range(class_n):
        ind =  [i for i in range(len(train_metric)) if train_lab[i] == k]
        tmp_ind = np.random.choice(ind, size=(args.sample_number, ),replace=True)
        train_loss_weight.append(train_metric[tmp_ind])
        
        train_ind.extend(tmp_ind)

elif args.test_mode == 'oracle_sampling_replacement':
    train_ind = []
    train_loss_weight = []
    for k in range(class_n):
        ind =  [i for i in range(len(train_metric)) if train_lab[i] == k]
        tmp_ind = np.random.choice(ind, size=(args.sample_number, ),replace=True, p=(train_metric[ind]+args.constant_1)/np.sum((train_metric[ind]+args.constant_1)))
        train_loss_weight.append(train_metric[tmp_ind])
        train_ind.extend(tmp_ind)
        
elif args.test_mode == 'oracle_sampling_replacement_mix':
    train_ind = []
    train_loss_weight = []
    for k in range(class_n):
        assert((args.constant_3>0) and (args.constant_3<1))
        ind =  [i for i in range(len(train_metric)) if train_lab[i] == k]
        len_1, len_2 = int(args.constant_3 * args.sample_number),  int((1-args.constant_3) * args.sample_number)
        tmp_ind_1 = np.random.choice(ind, size=(len_1, ),replace=True)
        tmp_ind_2 = np.random.choice(ind, size=(len_2, ),replace=True, p=(train_metric[ind]+args.constant_1)/np.sum((train_metric[ind]+args.constant_1)))
        # print(len(tmp_ind_1))
        # print(len(tmp_ind_2))
        tmp_ind = np.concatenate([tmp_ind_1,tmp_ind_2 ])
        # print(len(tmp_ind))
        # sss
        train_loss_weight.append(train_metric[tmp_ind])
        train_ind.extend(tmp_ind)

      
elif args.test_mode == 'pred_sampling':
    train_ind = []
    train_loss_weight = []
    for k in range(class_n):
        ind =  [i for i in range(len(train_metric)) if train_lab[i] == k]
        tmp_ind = np.random.choice(ind, size=(args.sample_number, ),replace=False, p=(train_metric[ind]+args.constant_1)/np.sum((train_metric[ind]+args.constant_1)))
        train_loss_weight.append(train_metric[tmp_ind])
        train_ind.extend(tmp_ind)
elif args.test_mode == "full":
    train_ind = []
    train_loss_weight = []
    for k in range(class_n):
        ind =  [i for i in range(len(train_metric)) if train_lab[i] == k]
        # tmp_ind = np.random.choice(ind, size=(args.sample_number, ),replace=False, p=(train_metric[ind]+args.constant_1)/np.sum((train_metric[ind]+args.constant_1)))
        train_loss_weight.append(train_metric[ind])
        train_ind.extend(ind)

# print(len(train_ind))
# print(len(np.unique(train_ind)))
# ssss

print(train_lab)
print(test_lab)
#####################################
train_data = train_data[train_ind]
train_lab = train_lab[train_ind]
################################
### vis un
if args.test_mode != 'coreset' and args.test_mode != 'badge':
    train_loss_weight = np.concatenate(train_loss_weight)
    print(len(train_loss_weight))

# import seaborn as sns 
# sns.set_theme()
# plt.hist(train_loss_weight, bins=30)
# plt.xlabel('Uncertainty Values')
# plt.ylabel('Instance Count')
# plt.savefig('sampling_v_un_woy.pdf')
# plt.close()
# ssss


###########################################

if  args.loss_type !=  "base":
    if args.loss_type ==  "reweight_clip_norm": 
        train_loss_weight = 1/train_loss_weight
        for k in range(10):
            train_loss_weight[k*args.sample_number:(k+1)*args.sample_number] /= np.mean(train_loss_weight[k*args.sample_number:(k+1)*args.sample_number])
        trainset = npy_dataset.NPY_Dataset_2(train_data, train_lab,train_loss_weight,  transform_train)
    else:
        trainset = npy_dataset.NPY_Dataset_2(train_data, train_lab,train_loss_weight,  transform_train)
else:
    trainset = npy_dataset.NPY_Dataset(train_data, train_lab, transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testset = npy_dataset.NPY_Dataset(train_data, train_lab, transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)
testset_2 = npy_dataset.NPY_Dataset(test_data, test_lab, transform_test)
testloader_2 = torch.utils.data.DataLoader(
    testset_2, batch_size=128, shuffle=False, num_workers=2)
testset_3 = npy_dataset.NPY_Dataset(real_test_data, real_test_lab, transform_test)
testloader_3 = torch.utils.data.DataLoader(
    testset_3, batch_size=128, shuffle=False, num_workers=2)

print('==> Building model..')


def train(epoch,  net,trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if  args.loss_type !=  "base": 
        criterion = nn.CrossEntropyLoss(reduction='none')
        for batch_idx, (inputs, targets,tr_weight) in enumerate(trainloader):
            inputs, targets, tr_weight = inputs.to(device), targets.to(device), tr_weight.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            t_loss = criterion(outputs, targets.long())
       #     pdb.set_trace()
            if args.loss_type ==  "reweight": 
                loss = torch.sum(t_loss/ (args.constant_2 +tr_weight )) / torch.sum(1/(args.constant_2 +tr_weight ))
            elif args.loss_type ==  "reweight_clip": 
                rw = torch.clip(1/tr_weight , max=args.constant_2)
                loss = torch.sum(t_loss * rw / torch.sum(rw))
            elif args.loss_type ==  "reweight_clip_norm": 
                rw = torch.clip(tr_weight , max=args.constant_2)
                loss = torch.sum(t_loss * rw / torch.sum(rw))
            # elif args.loss_type ==  "reweight_clip": 
            #     loss = torch.mean(t_loss * torch.clip(1/tr_weight , max=args.constant_2))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # _, predicted = outputs.max(1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Training Loss : ', train_loss /len(trainloader))
    else:
        criterion = nn.CrossEntropyLoss()
        for batch_idx, (inputs, targets,) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Training Loss : ', train_loss /len(trainloader))


def test(epoch, net,testloader, print_name):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    print(print_name, test_loss/len(testloader))
    print(print_name + ' Acc : ', acc)
    return acc


## 记录输出的logit

# torch.torch.manual_seed(0)

if args.model == 'resnet20':
    net = ResNet20(num_classes=class_n)
elif args.model == 'resnet32':
    net = ResNet32()
elif args.model == 'resnet44':
    net = ResNet44()
elif args.model == 'resnet56':
    net = ResNet56()
elif args.model == 'simplecnn':
    net = Vanillann()
elif args.model == 'mobilev2':
    net = MobileNetV2(num_classes=10)
elif args.model == 'dense121':
    net = DenseNet121()


net = net.to(device)


criterion = nn.CrossEntropyLoss()
if args.optimizer == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
elif args.optimizer == "adamw":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
if args.schedule_type == "step":
    scheduler =  torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=args.gamma)
elif args.schedule_type == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
for epoch in range(args.epochs):
    train(epoch,  net,trainloader)
    scheduler.step()
    test(epoch, net,testloader, 'Train Loss')
    test(epoch, net,testloader_2, 'Hold Test Loss')
    test(epoch, net,testloader_3, 'Real Test Loss')
