import torch
import torch.nn as nn
import torch.optim as optim
from utils.train_utils import get_pred_raw
import torchvision.transforms as transforms
import npy_dataset
import argparse
import numpy as np
import copy
from models.resnet import *
import os
import random
from models.simple_cnn import Vanillann
from models.densenet import DenseNet121
from models.mobilenet import MobileNetV2

def descent_lr_train(lr, epoch, optimizer, interval):
    for k in interval:
        if epoch < k:
            break
        else:
            lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--dataset', type=str, default='cifar10') 
parser.add_argument('--model', type=str, default='resnet20') 
parser.add_argument('--model_num', type=int, default=10) 
parser.add_argument('--num_classes', type=int, default=10) 
parser.add_argument('--epochs', type=int, default=100) 
parser.add_argument('--weight_folder', type=str, default='generation_weight_cifar10_resnet20/') 
parser.add_argument('--npy_folder', type=str, default='cifar10_npy_resnet20/') 
args = parser.parse_args()
os.makedirs(args.weight_folder, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.dataset == "cifar10":
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
    train_data =np.load("cifar_npy/train_X_1.npy")
    train_lab = np.load("cifar_npy/train_Y_1.npy")
    test_data =np.load( "cifar_npy/train_X_2.npy")
    test_lab = np.load("cifar_npy/train_Y_2.npy")
    real_test_data =np.load( "cifar_npy/test_X.npy")
    real_test_lab = np.load("cifar_npy/test_Y.npy")
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
    train_data =np.load("cifar100_npy/train_X_1.npy")
    train_lab = np.load("cifar100_npy/train_Y_1.npy")
    test_data =np.load( "cifar100_npy/train_X_2.npy")
    test_lab = np.load("cifar100_npy/train_Y_2.npy")
    real_test_data =np.load( "cifar100_npy/test_X.npy")
    real_test_lab = np.load("cifar100_npy/test_Y.npy")
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
    train_data =np.load("cifar_noise_npy/train_X_1.npy")
    train_lab = np.load("cifar_noise_npy/train_Y_1_aggre.npy")
    test_data =np.load( "cifar_noise_npy/train_X_2.npy")
    test_lab = np.load("cifar_noise_npy/train_Y_2_aggre.npy")
    real_test_data =np.load( "cifar_noise_npy/test_X.npy")
    real_test_lab = np.load("cifar_noise_npy/test_Y.npy")
    print(len(train_data), len(test_data), len(real_test_data))


trainset = npy_dataset.NPY_Dataset(train_data, train_lab, transform_train)
print(len(trainset))
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = npy_dataset.NPY_Dataset(train_data, train_lab, transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

testset_2 = npy_dataset.NPY_Dataset(test_data, test_lab, transform_test)
testloader_2 = torch.utils.data.DataLoader(
    testset_2, batch_size=128, shuffle=False, num_workers=2)

testset_3 = npy_dataset.NPY_Dataset(real_test_data, real_test_lab, transform_test)
testloader_3 = torch.utils.data.DataLoader(
    testset_3, batch_size=128, shuffle=False, num_workers=2)


# Model
print('==> Building model..')

os.makedirs(args.npy_folder, exist_ok=True)
def train(epoch,  net,trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
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


out_logit = np.zeros((args.model_num,len(train_lab), args.num_classes))
out_logit_2 = np.zeros((args.model_num,len(test_lab), args.num_classes))
out_logit_3 = np.zeros((args.model_num,len(real_test_lab), args.num_classes))

for k in range(args.model_num):
    setup_seed(k)
    torch.torch.manual_seed(k)
    if args.model == 'resnet20':
        net = ResNet20(num_classes=args.num_classes)
    elif args.model == 'resnet32':
        net = ResNet32(num_classes=args.num_classes)
    elif args.model == 'resnet44':
        net = ResNet44(num_classes=args.num_classes)
    elif args.model == 'resnet56':
        net = ResNet56(num_classes=args.num_classes)
    elif args.model == 'simplecnn':
        net = Vanillann()
    elif args.model == 'mobilev2':
        net = MobileNetV2(num_classes=args.num_classes)
    elif args.model == 'dense121':
        net = DenseNet121()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4 )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60, 90], gamma=0.1)
    for epoch in range(args.epochs):
        descent_lr_train(args.lr, epoch+1, optimizer, [30, 60 ,90])
        train(epoch, net, trainloader)
        test(epoch, net,testloader, 'Train Loss')
        test(epoch, net,testloader_2, 'Hold Test Loss')
        test(epoch, net,testloader_3, 'Real Test Loss')
    torch.save(net.state_dict(), os.path.join(args.weight_folder, 'uncertainty_generation_'+str(k)+'.pth'))
    tmp_logit = get_pred_raw(testloader, net, )
    out_logit[k] = copy.deepcopy(tmp_logit)
    tmp_logit_2 = get_pred_raw(testloader_2, net, )
    out_logit_2[k] = copy.deepcopy(tmp_logit_2)
    tmp_logit_3 = get_pred_raw(testloader_3, net, )
    out_logit_3[k] = copy.deepcopy(tmp_logit_3)

print(out_logit.shape)
np.save(os.path.join(args.npy_folder, "train_UN_1_raw.npy"), out_logit)
print(out_logit_2.shape)
np.save(os.path.join(args.npy_folder, "train_UN_2_raw.npy"), out_logit_2)
print(out_logit_3.shape)
np.save(os.path.join(args.npy_folder, "test_UN_raw.npy"), out_logit_3)
