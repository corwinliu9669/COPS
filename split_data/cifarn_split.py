import numpy as np
import pickle
import os
from PIL import Image
import copy
import torch
data_partition_seed=48
np.random.seed(data_partition_seed)
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    x = dict[b'data']
    x = np.reshape(x, [10000,3, 32,32])
    y = dict[b'labels']
    return x, y
noise_path = "raw_data/CIFAR-10_human.pt"

noise_label = torch.load(os.path.join(os.getcwd(), noise_path) )
clean_label = noise_label['clean_label'] 
worst_label = noise_label['worse_label'] 
aggre_label = noise_label['aggre_label'] 
random_label1 = noise_label['random_label1'] 
random_label2 = noise_label['random_label2'] 
random_label3 = noise_label['random_label3']
raw_data_path = "raw_data/cifar-10-batches-py"
save_path = "cifar_noise_npy"

os.makedirs(save_path, exist_ok=True)
train_file_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
test_file_list = ["test_batch"]

##############################################################
train_X, train_Y = [], []
test_X, test_Y = [], []
for tf in train_file_list:
    x,y= unpickle(os.path.join(raw_data_path, tf))
    train_X.append(copy.deepcopy(x))
    train_Y.append(copy.deepcopy(y))
train_X = np.concatenate(train_X, 0)
train_Y = np.concatenate(train_Y, 0)

# print(np.min(clean_label-train_Y))
# ssssss
print(train_X.shape)
print(train_Y.shape)
train_ind_whole = []
train_Y_1 = []
hold_ind_whole= []
train_Y_2 = []
for  j in range(10):
    ind_class = [k for k in range(len(train_Y)) if train_Y[k]==j]
    train_ind_whole.extend(ind_class[:1000])
    train_Y_1.extend([j]*1000)
    hold_ind_whole.extend(ind_class[1000: ])
    train_Y_2.extend([j]*4000)

train_X_1 = train_X[train_ind_whole]
train_X_2 = train_X[hold_ind_whole]
train_Y_1=np.array(train_Y_1)
train_Y_2=np.array(train_Y_2)

train_Y_1_wrost = worst_label[train_ind_whole]
train_Y_2_wrost = worst_label[hold_ind_whole]

train_Y_1_aggre = aggre_label[train_ind_whole]
train_Y_2_aggre = aggre_label[hold_ind_whole]

train_Y_1_rand1 = random_label1[train_ind_whole]
train_Y_2_rand1 = random_label1[hold_ind_whole]

train_Y_1_rand2 = random_label2[train_ind_whole]
train_Y_2_rand2 = random_label2[hold_ind_whole]

train_Y_1_rand3 = random_label3[train_ind_whole]
train_Y_2_rand3 = random_label3[hold_ind_whole]


# random_label1 = noise_label['random_label1'] 
# random_label2 = noise_label['random_label2'] 
# random_label3 = noise_label['random_label3']


print(train_X_1.shape)
print(train_Y_2_rand2.shape)

print(train_Y_2.shape)

print(train_X_2.shape)
print(train_Y_2.shape)

np.save(os.path.join(save_path, "train_X_1.npy"), train_X_1)
np.save(os.path.join(save_path, "train_Y_1.npy"), train_Y_1)
np.save(os.path.join(save_path, "train_X_2.npy"), train_X_2)
np.save(os.path.join(save_path, "train_Y_2.npy"), train_Y_2)

np.save(os.path.join(save_path, "train_Y_1_wrost.npy"), train_Y_1_wrost)
np.save(os.path.join(save_path, "train_Y_2_wrost.npy"), train_Y_2_wrost)

np.save(os.path.join(save_path, "train_Y_1_aggre.npy"), train_Y_1_aggre)
np.save(os.path.join(save_path, "train_Y_2_aggre.npy"), train_Y_2_aggre)

np.save(os.path.join(save_path, "train_Y_1_rand1.npy"), train_Y_1_rand1)
np.save(os.path.join(save_path, "train_Y_2_rand1.npy"), train_Y_2_rand1)

np.save(os.path.join(save_path, "train_Y_1_rand2.npy"), train_Y_1_rand2)
np.save(os.path.join(save_path, "train_Y_2_rand2.npy"), train_Y_2_rand2)

np.save(os.path.join(save_path, "train_Y_1_rand3.npy"), train_Y_1_rand3)
np.save(os.path.join(save_path, "train_Y_2_rand3.npy"), train_Y_2_rand3)





tf  = test_file_list[0]
x,y= unpickle(os.path.join(raw_data_path, tf))
test_X= x
test_Y= np.array(y)
print(test_X.shape)
print(test_Y.shape)
np.save(os.path.join(save_path, "test_X.npy"), test_X)
np.save(os.path.join(save_path, "test_Y.npy"), test_Y)




