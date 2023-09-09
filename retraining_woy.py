import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import npy_dataset
import argparse
import numpy as np
from models.resnet import *
import os
import random
from models.simple_cnn import Vanillann
from models.densenet import DenseNet121
from models.mobilenet import MobileNetV2

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', type=str, default='resnet20', help="name of save log") 
parser.add_argument('--optimizer', type=str, default='adamw', help="adamw/sgd") 
parser.add_argument('--dataset', type=str, default='cifar', help="cifar/svhn") 
parser.add_argument('--test_mode', type=str, default='random') 
parser.add_argument('--epochs', type=int, default=150) 
parser.add_argument('--T_max', type=int, default=50) 
parser.add_argument('--eta_min', default=0, type=float, help='min lr')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay')
parser.add_argument('--seed', type=int, default=1) 
parser.add_argument('--weight_decay', type=float, default=5e-4) 
parser.add_argument('--sample_number', type=int, default=300) 
parser.add_argument('--schedule_type', type=str, default="cosine", help="step/cosine/poly") 
parser.add_argument('--loss_type', type=str, default="base", help="base/reweight_clip") 
parser.add_argument('--weight_folder', type=str, default='active_weight/') 
parser.add_argument('--weight_name', type=str, default='active_weight.pth') 
parser.add_argument('--constant_1', type=float, default=0.005) 
parser.add_argument('--constant_2', type=float, default=0) 
parser.add_argument('--un_folder', type=str, default='cifar_npy') 
args = parser.parse_args()
pic_path = "./pics_" + args.dataset
os.makedirs(pic_path, exist_ok=True)
args.pic_name = os.path.join(pic_path, args.weight_name.replace(".pth", ".png"))

def cal_un_woy(logit):
    covariance = np.matmul((logit-np.mean(logit, axis=0, keepdims=True)).transpose()[1:, :], (logit-np.mean(logit, axis=0, keepdims=True))[:, 1:])/(len(logit)-1)
    
    logit =logit -np.max(logit, 1, keepdims=True)
    pred = np.mean(np.exp(logit)/np.sum(np.exp(logit), 1, keepdims=True), 0)[1:]

    pred_ = np.reshape(pred, (len(pred), 1))
    psi = np.diag(pred) - np.matmul(pred_, pred_.transpose())
    met = np.trace(np.matmul(psi, covariance))
    return met

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
setup_seed(args.seed)
os.makedirs(args.weight_folder, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.dataset == "cifar100":
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
    train_logit = np.load(os.path.join("cifar100_npy", "train_UN_2_raw.npy"))
    train_pseudo_label = np.argmax(np.mean(train_logit, 0), 1)
    print(train_pseudo_label)
    for j in range(100):
        print(len([i for i in range(len(train_pseudo_label)) if train_pseudo_label[i]==j]))

    train_logit = train_logit.transpose((1,0,2))
    train_metric = np.zeros((len(train_logit), ))
    for k in range(len(train_logit)):
        train_metric[k] = cal_un_woy(train_logit[k] )


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
    train_logit = np.load(os.path.join('cifar100_npy', "train_UN_2_raw.npy"))
    train_pseudo_label = np.argmax(np.mean(train_logit, 0), 1)
    print(train_pseudo_label)
    for j in range(10):
        print(len([i for i in range(len(train_pseudo_label)) if train_pseudo_label[i]==j]))

    train_logit = train_logit.transpose((1,0,2))
    train_metric = np.zeros((len(train_logit), ))
    for k in range(len(train_logit)):
        train_metric[k] = cal_un_woy(train_logit[k] )



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
    train_logit = np.load(os.path.join("cifar_aggre_npy", "train_UN_2_raw.npy"))
    train_logit = train_logit.transpose((1,0,2))
    train_logit = train_logit.transpose((1,0,2))
    train_metric = np.zeros((len(train_logit), ))
    for k in range(len(train_logit)):
        train_metric[k] = cal_un_woy(train_logit[k] )
    
######################### sample_selection
train_metric[train_metric<=0] = np.min(train_metric[train_metric>0])
train_metric = np.sqrt(train_metric)



######################### sample_selection

if args.test_mode == 'random':
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

    ind =  [i for i in range(len(train_metric)) ]
    tmp_ind = np.random.choice(ind, size=(args.sample_number*class_n, ),replace=False, p=(train_metric_[ind])/np.sum((train_metric_[ind])))
    train_loss_weight.append(train_metric[tmp_ind])
    train_ind.extend(tmp_ind)





#####################################
train_data = train_data[train_ind]
train_lab = train_lab[train_ind]
################################


###########################################

if  args.loss_type !=  "base":

    trainset = npy_dataset.NPY_Dataset_2(train_data, train_lab,train_loss_weight,  transform_train)
else:
    trainset = npy_dataset.NPY_Dataset(train_data, train_lab, transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=256, shuffle=True, num_workers=2)
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
            if args.loss_type ==  "reweight_clip": 
                rw = torch.clip(1/tr_weight , max=args.constant_2)
                loss = torch.sum(t_loss * rw / torch.sum(rw))
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
    net = ResNet32(num_classes=class_n)
elif args.model == 'resnet44':
    net = ResNet44(num_classes=class_n)
elif args.model == 'resnet56':
    net = ResNet56(num_classes=class_n)
elif args.model == 'simplecnn':
    net = Vanillann()
elif args.model == 'mobilev2':
    net = MobileNetV2(num_classes=class_n)
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
    test(epoch, net,testloader_3, 'Real Test Loss')
