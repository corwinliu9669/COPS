from torch.utils.data import Dataset
from PIL import Image
class NPY_Dataset(Dataset):
    def __init__(self, data, lab, transform):
        self.data = data
        self.lab = lab
        self.data_len = len(data)
        self.transform= transform
        assert(len(data) == len(lab))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        sample_data = self.data[index]
        # print(sample_data.shape)
        sample_data = self.transform(Image.fromarray(sample_data.transpose()))
        sample_lab = self.lab[index]
        return (sample_data, sample_lab)

class NPY_Dataset_2(Dataset):
    def __init__(self, data, lab, weight, transform):
        self.data = data
        self.lab = lab
        self.weight = weight
        self.data_len = len(data)
        self.transform= transform
        assert(len(data) == len(lab))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        sample_data = self.data[index]
        # print(sample_data.shape)
        sample_data = self.transform(Image.fromarray(sample_data.transpose()))
        sample_lab = self.lab[index]
        sample_weight = self.weight[index]
        return (sample_data, sample_lab, sample_weight)