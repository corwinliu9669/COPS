import torch
import torch.nn as nn
import torch.optim as optim
from models.gru import BiGRU
import argparse
import numpy as np
from torchtext.data.utils import get_tokenizer
from npy_dataset_imdb import NPY_Dataset, NPY_Dataset_2
import os
import random
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator 
from torch.utils.data import DataLoader
import random
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--model', type=str, default='gru', help="name of save log") 
parser.add_argument('--optimizer', type=str, default='adamw', help="adamw/sgd") 
parser.add_argument('--dataset', type=str, default='imdb', help="cifar/svhn") 
parser.add_argument('--test_mode', type=str, default='oracle_sampling', help="oracle_sampling/random/predict/full") 
parser.add_argument('--epochs', type=int, default=20) 
parser.add_argument('--T_max', type=int, default=50) 
parser.add_argument('--eta_min', default=0, type=float, help='min lr')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay')
parser.add_argument('--seed', type=int, default=1) 
parser.add_argument('--weight_decay', type=float, default=5e-4) 
parser.add_argument('--sample_number', type=int, default=7500) 
parser.add_argument('--schedule_type', type=str, default="cosine", help="step/cosine/poly") 
parser.add_argument('--loss_type', type=str, default="reweight", help="base/reweight/reweight_clip") 
parser.add_argument('--weight_folder', type=str, default='active_weight_binary/') 
parser.add_argument('--weight_name', type=str, default='active_weight.pth') 
parser.add_argument('--constant_1', type=float, default=0) 
parser.add_argument('--constant_2', type=float, default=0.3) 
parser.add_argument('--un_folder', type=str, default='imdb_npy_bert_sent') 
parser.add_argument('--un_type', type=str, default='woy',help='woy/wy') 
args = parser.parse_args()

def with_y_un(train_logit , train_lab):
    train_std= np.std(train_logit, axis=0)
    train_std = np.sum(train_std, axis=-1)
    train_mean = np.mean(np.exp(train_logit)/(1+np.exp(train_logit)),  axis=0)
    train_metric = np.abs(train_mean.ravel()-train_lab.ravel())*train_std
    return train_metric

def wo_y_un(train_logit):
    train_std= np.std(train_logit, axis=0)
    train_std = np.sum(train_std, axis=-1)
    train_mean = np.mean(np.exp(train_logit)/(1+np.exp(train_logit)),  axis=0)
    train_mean = np.sum(train_mean, axis=-1)
    train_metric = np.sqrt(train_mean-train_mean*train_mean) * train_std
    return train_metric

def get_lab(train_list):
    lab_list = []
    n_1 = 0
    n_2 = 0
    for i ,(la,_) in enumerate(train_list):
        lab_list.append(la)
        # print(la)
        if la==1:
            n_1 +=1
        if la==2:
            n_2 +=1
    # print(n_1, n_2)
    return np.array(lab_list)-1
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(args.seed)
os.makedirs(args.weight_folder, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = get_tokenizer("basic_english") 
def yield_tokens(data_iter): 
   for _, text in data_iter: 
       yield tokenizer(text) 

def get_vocab(train_datapipe): 
    vocab = build_vocab_from_iterator(yield_tokens(train_datapipe), 
                                     specials=['<UNK>', '<PAD>'], 
                                    max_tokens=2000, min_freq=30) 
    vocab.set_default_index(vocab['<UNK>']) 
    return vocab 

def transform_text(x):
    return vocab(tokenizer(x))

if args.dataset == "imdb":
    train_iter = IMDB(split='train')
    vocab = get_vocab(train_iter)
    whole_train_set = list(train_iter)
    test_iter = IMDB(split='test')
    test_list = list(test_iter)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True, max_length=512)
    random.seed(1)
    random.shuffle(whole_train_set)
    random.seed(2)
    random.shuffle(whole_train_set)
    hold_train_list, train_list = whole_train_set[:5000], whole_train_set[5000:]
    train_lab=get_lab(train_list)
    test_lab=get_lab(hold_train_list)
    real_test_lab=get_lab(test_list)
    train_logit = np.load(os.path.join(args.un_folder, "train_UN_2_raw.npy"))
    if args.un_type == 'wy':
        train_metric=with_y_un(train_logit , train_lab)
    elif args.un_type == 'woy':
        train_metric=wo_y_un(train_logit )

######################### sample_selection
setup_seed(args.seed)
if args.test_mode == 'random':
    train_ind = []
    train_loss_weight = []
    for k in range(2):
        ind =  [i for i in range(len(train_metric)) if train_lab[i] == k]
        tmp_ind = np.random.choice(ind, size=(args.sample_number, ),replace=False)
        train_loss_weight.append(train_metric[tmp_ind])
        train_ind.extend(tmp_ind)


if args.test_mode == 'random_whole':
    train_ind = []
    train_loss_weight = []
    # for k in range(10):
    ind =  [i for i in range(len(train_metric))]
    tmp_ind = np.random.choice(ind, size=(args.sample_number*2, ),replace=False)
    train_loss_weight.append(train_metric[tmp_ind])
    train_ind.extend(tmp_ind)
elif args.test_mode == 'oracle_sampling_cut':
    train_metric_ = np.clip(train_metric,a_min=0, a_max=args.constant_1)
    train_ind = []
    train_loss_weight = []
    for k in range(2):
        ind =  [i for i in range(len(train_metric)) if train_lab[i] == k]
        tmp_ind = np.random.choice(ind, size=(args.sample_number, ),replace=False, p=(train_metric_[ind])/np.sum((train_metric_[ind])))
        train_loss_weight.append(train_metric[tmp_ind])
        train_ind.extend(tmp_ind)
elif args.test_mode == 'oracle_sampling_cut_whole':
    train_metric_ = np.clip(train_metric,a_min=0, a_max=args.constant_1)
    train_ind = []
    train_loss_weight = []

    ind =  [i for i in range(len(train_metric)) ]
    tmp_ind = np.random.choice(ind, size=(args.sample_number*2, ),replace=False, p=(train_metric_[ind])/np.sum((train_metric_[ind])))
    train_loss_weight.append(train_metric[tmp_ind])
    train_ind.extend(tmp_ind)

train_data = []
for i in train_ind:
    train_data.append(train_list[i])
################################


train_loss_weight = np.concatenate(train_loss_weight)
print(len(train_loss_weight))
print(len(train_data ))

###########################################

if  args.loss_type !=  "base":
    train_set = NPY_Dataset_2(train_data, transform_text, 256, train_loss_weight)
else:
    train_set = NPY_Dataset(train_data, transform_text,max_length=256)
trainloader = DataLoader(train_set,batch_size=32, shuffle=True)

train_test_set = NPY_Dataset(train_data, transform_text,max_length=256)

hold_train = NPY_Dataset(hold_train_list, transform_text,max_length=256)
test_set = NPY_Dataset(test_list, transform_text,max_length=256)

testloader = DataLoader(train_test_set,batch_size=256, shuffle=False)
testloader_2 = DataLoader(hold_train,batch_size=256, shuffle=False)
testloader_3 = DataLoader(test_set,batch_size=256, shuffle=False)


print('==> Building model..')


def train(epoch,  net,trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if  args.loss_type !=  "base": 
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        for batch_idx, (inputs, targets,tr_weight) in enumerate(trainloader):
            inputs, targets, tr_weight = inputs.to(device).long(), targets.to(device), tr_weight.to(device)
            optimizer.zero_grad()
            outputs = net(inputs).view(-1)
            t_loss = criterion(outputs, targets.float())
            if args.loss_type ==  "reweight": 
                loss = torch.sum(t_loss/ (args.constant_2 +tr_weight )) / torch.sum(1/(args.constant_2 +tr_weight ))
            elif args.loss_type ==  "reweight_clip": 
                rw = torch.clip(1/tr_weight , max=args.constant_2)
                loss = torch.sum(t_loss * rw / torch.sum(rw))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += targets.size(0)
            correct += (torch.sigmoid(outputs).ge(0.5) == targets).sum().item()
        print('Training Loss : ', train_loss /len(trainloader))
    else:
        criterion = nn.BCEWithLogitsLoss()
        for batch_idx, (inputs, targets,) in enumerate(trainloader):
            inputs, targets = inputs.to(device).long(), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs).view(-1)
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += targets.size(0)
            correct += (torch.sigmoid(outputs).ge(0.5) == targets).sum().item()
        print('Training Loss : ', train_loss /len(trainloader))


def test(epoch, net,testloader, print_name):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device).long(), targets.to(device)
            outputs = net(inputs).view(-1)
            loss = criterion(outputs, targets.float())
            test_loss += loss.item()
            total += targets.size(0)
            correct += (torch.sigmoid(outputs).ge(0.5) == targets).sum().item()
    acc = 100.*correct/total
    print(print_name, test_loss/len(testloader))
    print(print_name + ' Acc : ', acc)
    return acc

if args.model=='gru':
    net= BiGRU()
    net = net.to(device)

criterion = nn.BCEWithLogitsLoss()
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
    test(epoch, net,testloader, 'Train Loss')
    test(epoch, net,testloader_3, 'Real Test Loss')
