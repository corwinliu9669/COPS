import torch
import torch.nn as nn
import torch.optim as optim
from npy_dataset_imdb import NPY_Dataset
import argparse
from models.gru import BiGRU
import numpy as np
import copy
import os
import random
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator 



def get_pred_logit_presig(testloader, net, device='cuda' ):
    ##### here we record the variance of each prediction 
    net.eval()
    logit_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device).long(), targets.to(device)
            outputs = net(inputs)
            out =outputs.detach().cpu().numpy()
            logit_list.append(out)
        logit_list=np.concatenate(logit_list, 0)
        print(logit_list.shape)
    return logit_list

def descent_lr_train(lr, epoch, optimizer, interval):
    for k in interval:
        if epoch < k:
            break
        else:
            lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('***********************************')
    print('learning rate:', lr)
    print('***********************************')

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
    print(n_1, n_2)
    return np.array(lab_list)-1
def get_data(train_list):
    lab_list = []

    for i ,(_,la) in enumerate(train_list):
        lab_list.append(la)

    return lab_list
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--model', type=str, default='gru') 
parser.add_argument('--model_num', type=int, default=10) 
parser.add_argument('--epochs', type=int, default=20) 
parser.add_argument('--weight_folder', type=str, default='generation_weight_binary_imdb/') 
parser.add_argument('--npy_folder', type=str, default='imdb_npy/') 
args = parser.parse_args()
os.makedirs(args.weight_folder, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


os.makedirs(args.npy_folder, exist_ok=True)


def train(epoch,  net,trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
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
train_iter = IMDB(split='train')
whole_train_set = list(train_iter)
test_iter = IMDB(split='test')
test_list = list(test_iter)

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
vocab = get_vocab(train_iter)
def transform_text(x):
    return vocab(tokenizer(x))

random.seed(1)
random.shuffle(whole_train_set)
random.seed(2)
random.shuffle(whole_train_set)
train_list, hold_train_list = whole_train_set[:5000], whole_train_set[5000:]
train_lab=get_lab(train_list)
test_lab=get_lab(hold_train_list)
real_test_lab=get_lab(test_list)


train_set = NPY_Dataset(train_list, transform_text,max_length=256)
hold_train = NPY_Dataset(hold_train_list, transform_text,max_length=256)
test_set = NPY_Dataset(test_list, transform_text,max_length=200)
trainloader = DataLoader(train_set,batch_size=32, shuffle=True)
testloader = DataLoader(train_set,batch_size=256, shuffle=False)
testloader_2 = DataLoader(hold_train,batch_size=256, shuffle=False)
testloader_3 = DataLoader(test_set,batch_size=256, shuffle=False)

out_logit = np.zeros((args.model_num,len(train_set), 1))
out_logit_2 = np.zeros((args.model_num,len(hold_train), 1))
out_logit_3 = np.zeros((args.model_num,len(test_set), 1))

for k in range(args.model_num):
    setup_seed(k)
    if args.model=='gru':
        model= BiGRU()
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        train(epoch, model, trainloader)
        test(epoch, model,testloader, 'Train Loss')
        test(epoch, model,testloader_2, 'Hold Test Loss')
        test(epoch, model,testloader_3, 'Real Test Loss')
    torch.save(model.state_dict(), os.path.join(args.weight_folder, 'uncertainty_generation_'+str(k)+'.pth'))
    tmp_logit = get_pred_logit_presig(testloader, model, )
    out_logit[k] = copy.deepcopy(tmp_logit)
    tmp_logit_2 = get_pred_logit_presig(testloader_2, model, )
    out_logit_2[k] = copy.deepcopy(tmp_logit_2)
    tmp_logit_3 = get_pred_logit_presig(testloader_3, model, )
    out_logit_3[k] = copy.deepcopy(tmp_logit_3)
    
print(out_logit.shape)
np.save( os.path.join(args.npy_folder, "train_UN_1_raw.npy"), out_logit)
print(out_logit_2.shape)
np.save( os.path.join(args.npy_folder, "train_UN_2_raw.npy"), out_logit_2)
print(out_logit_3.shape)
np.save( os.path.join(args.npy_folder, "test_UN_raw.npy"), out_logit_3)
