import numpy as np
import pickle
import os
from PIL import Image
import copy
data_partition_seed=48
np.random.seed(data_partition_seed)
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    # print(dict.keys())

    x = dict[b'data']
    x = np.reshape(x, [len(x),3, 32,32])
    y = np.array(dict[b'fine_labels'])
 #   print(y)
 #   print(x.shape)
 #   print(y.shape)
    # print(np.unique(y))
    return x, y
# print(os.getcwd())
raw_data_path = "raw_data/cifar-100-python/"
save_path = "cifar100_npy"
#print(unpickle(raw_data_path+'train'))
#ssss

os.makedirs(save_path, exist_ok=True)
train_file_list = ["train"]
test_file_list = ["test"]

##############################################################
train_X, train_Y = [], []
test_X, test_Y = [], []
for tf in train_file_list:
    x,y= unpickle(os.path.join(raw_data_path, tf))
    train_X.append(copy.deepcopy(x))
    train_Y.append(copy.deepcopy(y))
train_X = np.concatenate(train_X, 0)
train_Y = np.concatenate(train_Y, 0)
# np.save('cifar_100_or.npy', train_Y)

# print(train_X.shape)
# print(train_Y.shape)
train_ind_whole = []
train_Y_1 = []
hold_ind_whole= []
train_Y_2 = []
for  j in range(100):
    ind_class = [k for k in range(len(train_Y)) if train_Y[k]==j]
    train_ind_whole.extend(ind_class[:200])
    train_Y_1.extend([j]*200)
    hold_ind_whole.extend(ind_class[200: ])
    train_Y_2.extend([j]*300)
    # print(len(ind_class))
#ssss
train_X_1 = train_X[train_ind_whole]
train_X_2 = train_X[hold_ind_whole]
train_Y_1=np.array(train_Y_1)
train_Y_2=np.array(train_Y_2)
# print(train_X_1.shape)
# print(train_Y_1.shape)
# print(train_X_2.shape)
# print(train_Y_2.shape)

np.save(os.path.join(save_path, "train_X_1.npy"), train_X_1)
np.save(os.path.join(save_path, "train_Y_1.npy"), train_Y_1)
np.save(os.path.join(save_path, "train_X_2.npy"), train_X_2)
np.save(os.path.join(save_path, "train_Y_2.npy"), train_Y_2)


tf  = test_file_list[0]
x,y= unpickle(os.path.join(raw_data_path, tf))
test_X= x
test_Y= np.array(y)
# print(test_X.shape)
# print(test_Y.shape)
np.save(os.path.join(save_path, "test_X.npy"), test_X)
np.save(os.path.join(save_path, "test_Y.npy"), test_Y)




