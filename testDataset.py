from operator import index
import pandas as pd
import os

from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
class myDataset(Dataset):
    def __init__(self, root_dir='dataset/ETT-small/', filename='ETTh1.csv'):
        self.root_dir = root_dir
        self.filename = filename
        self.total_path = os.path.join(self.root_dir, self.filename)
        self.raw_data = pd.read_csv(self.total_path)
    def __len__(self):
        # return self.raw_data.size
        return len(self.raw_data)
    def __getitem__(self, index=1):
        data = self.raw_data.iloc[:, 1:].values.astype('float32')
        # data = self.raw_data.iloc[:, index:index+1].values.astype('float32')
        data = torch.tensor(data)
        return data

mydataset = myDataset()
# print(mydataset[0])
# print(len(mydataset))
dataloader = DataLoader(mydataset, batch_size=1, shuffle=False)
# for i, (date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT) in enumerate(dataloader):
for i, data in enumerate(dataloader):
    print(i, data)
    print(data.shape)