import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from scipy.io import loadmat
import h5py

import numpy as np
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device1 = "cuda:1" if torch.cuda.is_available() else "cpu"
device2 = "cuda:2" if torch.cuda.is_available() else "cpu"
device3 = "cuda:3" if torch.cuda.is_available() else "cpu"

class DJIGetData2D(Dataset):
    """
    return data for dataloader
    """
    def __init__(self, trainingset=['5G_direct_Coe', '5G_excite_OneHot'], device=device1):
        """输入数据的1-200000用来训练和验证"""
        Y = h5py.File(f'./data/{trainingset[0]}.mat')
        Y = np.transpose(Y['direct_Coe'])[0:50000,:,:]
        self.direction = torch.tensor(Y, dtype=torch.float32)
        # self.direction = self.direction.permute(0, 2, 1)
        self.direction.unsqueeze_(dim=1)
        # self.amp = torch.max(torch.max(torch.abs(self.direction), dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
        # self.direction = self.direction / self.amp
        self.device = device

        X = loadmat(f'./data/{trainingset[1]}.mat')
        # X = h5py.File(f'./data/Training_Validing/{trainingset[1]}.mat')
        X = np.array(X['excite_OneHot'])[0:50000,:,:]

        self.target = torch.tensor(X, dtype=torch.float32)
        # self.target.transpose(0,2,1)
        # self.target = self.target.view(-1, 2, 3, 3)
        # print("a")

    def __getitem__(self, item):
        data = {}
        # data['echo'] = self.direction[item].to(self.device)
        # data['amp'] = self.amp[item].to(self.device)
        data['echo'] = self.direction[item].to(self.device)
        data['target'] = self.target[item].to(self.device)
        return data

    def __len__(self):
        return len(self.direction)

class DJIGetTestData2D(Dataset):
    """
    return data for dataloader
    """
    def __init__(self, trainingset=['5G_direct_Coe', '5G_excite_OneHot'], device=device1):
        """输入数据的1-200000用来训练和验证"""
        Y = h5py.File(f'./data/{trainingset[0]}.mat')
        Y = np.transpose(Y['direct_Coe'])[0:5, :, :]
        self.direction = torch.tensor(Y, dtype=torch.float32)
        # self.direction = self.direction.permute(0, 2, 1)
        self.direction.unsqueeze_(dim=1)
        # self.amp = torch.max(torch.max(torch.abs(self.direction), dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
        # self.direction = self.direction / self.amp
        self.device = device

        X = loadmat(f'./data/{trainingset[1]}.mat')
        # X = h5py.File(f'./data/Training_Validing/{trainingset[1]}.mat')
        X = np.array(X['excite_OneHot'])[0:5, :, :]

        self.target = torch.tensor(X, dtype=torch.float32)

    def __getitem__(self, item):
        data = {}
        data['echo'] = self.direction[item].to(self.device)
        data['target'] = self.target[item].to(self.device)
        return data

    def __len__(self):
        return len(self.direction)

class DJIGetSingleTestData2D(Dataset):

    def __init__(self, trainingset=['direct_Coe_30', 'excite_OneHot_30'], device=device1):
        Y = h5py.File(f'./data/Direct/{trainingset[0]}.mat')
        # print(Y.keys())
        Y = np.transpose(Y['direct_Coe_Total'])[0:20, :, :]
        self.direction = torch.tensor(Y, dtype=torch.float32)
        # self.direction = self.direction.permute(0, 2, 1)
        self.direction.unsqueeze_(dim=1)
        self.device = device

        X = h5py.File(f'./data/Excite/{trainingset[1]}.mat')
        # print(X.keys())
        # X = h5py.File(f'./data/Training_Validing/{trainingset[1]}.mat')
        X = np.transpose(X['excite_OneHot_Total'])[0:20, :, :]

        self.target = torch.tensor(X, dtype=torch.float32)

    def __getitem__(self, item):
        data = {}
        data['echo'] = self.direction[item].to(self.device)
        data['target'] = self.target[item].to(self.device)
        return data

    def __len__(self):
        return len(self.direction)



class DJIGetData(Dataset):
    """
    return data for dataloader
    """
    def __init__(self, trainingset=['Farfield-121', 'DC'], device=device1):
        """输入数据的1-300000用来训练和验证"""
        Y = loadmat(f'./data/Training_Validing/{trainingset[0]}.mat')
        # Y = h5py.File(f'./data/Training_Validing/{trainingset[0]}.mat')
        Y = np.array(Y['Pattern3'])[0:50000]
        self.direction = torch.tensor(Y, dtype=torch.float32)
        self.direction = self.direction.view(-1, 1, 11, 11)
        print(self.direction.shape)

        X = loadmat(f'./data/Training_Validing/{trainingset[1]}.mat')
        # X = h5py.File(f'./data/Training_Validing/{trainingset[1]}.mat')
        X = np.array(X['DC'])[0:50000]
        self.target = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        # self.target.transpose(0,2,1)
        self.target = self.target.view(-1, 2, 3, 3)
        print(self.target.shape)
        self.device = device

    def __getitem__(self, item):
        data = {}
        data['echo'] = self.direction[item].to(self.device)
        data['target'] = self.target[item].to(self.device)
        return data

    def __len__(self):
        return len(self.direction)

class DJIGetData2(Dataset):
    """
    return data for dataloader
    """
    def __init__(self, trainingset=['Farfield-121', 'DC'], device=device1):
        """输入数据的1-300000用来训练和验证"""
        Y = loadmat(f'./data/Training_Validing/{trainingset[0]}.mat')
        # Y = h5py.File(f'./data/Training_Validing/{trainingset[0]}.mat')
        Y = np.array(Y['Pattern3'])[0:300000]
        self.direction = torch.tensor(Y, dtype=torch.float32)
        self.direction = self.direction.view(-1, 1, 11, 11)
        print(self.direction.shape)

        X = loadmat(f'./data/Training_Validing/{trainingset[1]}.mat')
        # X = h5py.File(f'./data/Training_Validing/{trainingset[1]}.mat')
        X = np.array(X['DC'])[0:300000]
        self.target = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        # self.target.transpose(0,2,1)
        self.target = self.target.view(-1, 2, 3, 3)
        print(self.target.shape)
        self.device = device

    def __getitem__(self, item):
        data = {}
        data['echo'] = self.direction[item].to(self.device)
        data['target'] = self.target[item].to(self.device)
        return data

    def __len__(self):
        return len(self.direction)


class DJIGetTestData(Dataset):
    """
    return data for dataloader
    """
    def __init__(self, trainingset=['Farfield-121', 'DC'], device=device1):
        """输入数据的1-300000用来训练和验证"""
        Y = loadmat(f'./data/Training_Validing/{trainingset[0]}.mat')
        # Y = h5py.File(f'./data/Training_Validing/{trainingset[0]}.mat')
        Y = np.array(Y['Pattern3'])[1:2]
        self.direction = torch.tensor(Y, dtype=torch.float32)
        self.direction = self.direction.view(-1, 1, 11, 11)
        print(self.direction.shape)
        print(self.direction[0])

        X = loadmat(f'./data/Training_Validing/{trainingset[1]}.mat')
        X = np.array(X['DC'])[1:2]
        self.target = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        # self.target.transpose(0,2,1)
        self.target = self.target.view(-1, 2, 3, 3)
        print(self.target.shape)
        print(self.target[0])
        self.device = device

    def __getitem__(self, item):
        data = {}
        data['echo'] = self.direction[item].to(self.device)
        data['target'] = self.target[item].to(self.device)
        return data

    def __len__(self):
        return len(self.direction)

