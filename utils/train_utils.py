import torch
import numpy as np


def get_pred_logit(testloader, net, device='cuda' ):
    ##### here we record the variance of each prediction 
    net.eval()
    logit_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            out = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            logit_list.append(out)
        logit_list=np.concatenate(logit_list, 0)
        print(logit_list.shape)
    return logit_list
def get_pred_raw(testloader, net, device='cuda' ):
    ##### here we record the variance of each prediction 
    net.eval()
    logit_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            out = outputs.detach().cpu().numpy()
            logit_list.append(out)
        logit_list=np.concatenate(logit_list, 0)
        print(logit_list.shape)
    return logit_list

def get_pred_un_sig(testloader, net, device='cuda' ):
    ##### here we record the variance of each prediction 
    net.eval()
    logit_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            out =  torch.sigmoid(outputs).detach().cpu().numpy()
            logit_list.append(out)
        logit_list=np.concatenate(logit_list, 0)
        print(logit_list.shape)
    return logit_list

def get_pred_un(testloader, net, device='cuda' ):
    ##### here we record the variance of each prediction 
    net.eval()
    logit_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            out =  torch.sigmoid(outputs).detach().cpu().numpy()
            logit_list.append(out)
        logit_list=np.concatenate(logit_list, 0)
        print(logit_list.shape)
    return logit_list
