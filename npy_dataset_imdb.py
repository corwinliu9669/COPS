import torch
from torch.utils.data import Dataset
import numpy as np
class NPY_Dataset(Dataset):
    def __init__(self, data, token,max_length):
        self.data = data
        self.data_len = len(data)
        self.token= token
        self.max_length =max_length
   

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        sample_data = self.data[index]
        text = sample_data[1]
        # text = "[CLS] " +text
        text = self.token(text)
        if len(text) > self.max_length:
             text = text[:self.max_length]
        sample_lab = sample_data[0]-1
        if len(text) <= self.max_length:
            text = np.concatenate([np.array(text).ravel(), np.zeros((self.max_length-len(text),)).ravel()])
        return (torch.Tensor(text), sample_lab)

class NPY_Dataset_2(Dataset):
    def __init__(self, data, token,max_length, un):
        self.data = data
        self.data_len = len(data)
        self.token= token
        self.max_length =max_length
        self.un = un
   

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        sample_data = self.data[index]
        text = sample_data[1]
        text = self.token(text)
        if len(text) > self.max_length:
             text = text[:self.max_length]
        sample_lab = sample_data[0]-1
        if len(text) <= self.max_length:
            text = np.concatenate([np.array(text).ravel(), np.zeros((self.max_length-len(text),)).ravel()])
        un_sample = self.un[index]
        return (torch.Tensor(text), sample_lab, un_sample)