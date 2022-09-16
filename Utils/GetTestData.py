import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from scipy.io import loadmat
device = "cpu"
# "cuda:2" if torch.cuda.is_available() else
class GetTestData(Dataset):
    """
    Return data for the dataloader.
    """
    def __init__(self):
        # R_y = loadmat('E:/WangJianyang/MatLabPro/Sparse Aperture/test/D/ship1/test_Y_real.mat')
        # I_y = loadmat('E:/WangJianyang/MatLabPro/Sparse Aperture/test/D/ship1/test_Y_imag.mat')
        R_y = loadmat('./Data/test/D/Yak42/Yak42_Y_real.mat')
        I_y = loadmat('./Data/test/D/Yak42/Yak42_Y_imag.mat')
        self.R_y = self.trans(R_y['Yak42_Y_real']).squeeze().to(dtype=torch.float32, device=device)
        self.I_y = self.trans(I_y['Yak42_Y_imag']).squeeze().to(dtype=torch.float32, device=device)
        self.length = len(self.R_y)

    def __getitem__(self, item):
        data = {}
        data['test_Y_real'] = self.R_y[item]
        data['test_Y_imag'] = self.I_y[item]
        return data
    # {'Y_real':self.R_y[item], 'Y_imag':self.I_y[item], 'X_real':self.R_x[item], 'X_imag':self.I_x[item]}

    def __len__(self):
        return self.length

    def trans(self, data):
        trans = transforms.Compose([
            transforms.ToTensor()
        ])
        return trans(data)


class GetTestData1(Dataset):
    """
    return data for dataloader
    """
    def __init__(self):
        R_y = loadmat('./Data/Train_10dB_10000/echo_real_10dB.mat')
        I_y = loadmat('./Data/Train_10dB_10000/echo_img_10dB.mat')
        self.R_y = self.trans(R_y['echo_save_real']).squeeze().to(dtype=torch.float32, device=device)
        self.I_y = self.trans(I_y['echo_save_img']).squeeze().to(dtype=torch.float32, device=device)
        self.length = len(self.R_y)

    def __getitem__(self, item):
        data = {}
        data['test_Y_real_other'] = self.R_y[item]
        data['test_Y_imag_other'] = self.I_y[item]
        return data
    # {'Y_real':self.R_y[item], 'Y_imag':self.I_y[item], 'X_real':self.R_x[item], 'X_imag':self.I_x[item]}

    def __len__(self):
        return self.length

    def trans(self, data):
        trans = transforms.Compose([
            transforms.ToTensor()
        ])
        return trans(data)