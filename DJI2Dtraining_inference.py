import sys
import datetime
import torch
import torch.nn as nn

from Utils.GetData import DJIGetData2D, DJIGetTestData2D, DJIGetSingleTestData2D
from torch.utils.data import DataLoader, random_split
from Model.DJImodule import DJI2D
from torch.utils.tensorboard import SummaryWriter

from Utils.makefile import makefile
from Utils.Time import Timer
from Utils.Train import DJI2DTrainOnlyAmp
from matplotlib import pyplot as plt
"""config information"""
cuda0 = "cuda:0" if torch.cuda.is_available() else "cpu"
cuda1 = "cuda:1" if torch.cuda.is_available() else "cpu"
cuda2 = "cuda:2" if torch.cuda.is_available() else "cpu"
cuda3 = "cuda:3" if torch.cuda.is_available() else "cpu"

# batch_size = 512
# epoch_num = 300
# dataset_num = 25000
# train_valid_rate = 0.9
# warmup_step = 400
# learning_rate = 0.001
# weight_decay = 1e-3

batch_size = 128
epoch_num = 500
dataset_num = 50000
train_valid_rate = 0.95
warmup_step = 4000
learning_rate = 0.001
weight_decay = 1e-5

"""-----------------------------Filename of TrainLog and Param vary with parameters ---------------------------"""
paramfilename = f'DJI2D_onehot CrossEntropy warmup{warmup_step} lr{learning_rate} dataset{dataset_num} bs{batch_size} epoch{epoch_num}'
#
# """hyper-parameters"""
# batch_size = 512
# epoch_num = 300
# dataset_num = 25000
# train_valid_rate = 0.9
#
# warmup_step = 400
# # embedding_dim = 30
# learning_rate = 0.001
# weight_decay = 1e-3
# # pulse = 10
#
# """These configs must be modified every time."""
# paramfilename = f'DJI2D_AP Y_X NMSE warmup{warmup_step} lr{learning_rate} dataset{dataset_num} bs{batch_size} epoch{epoch_num}'
Train_flag = False

time = Timer()
# device = "cuda:1"
device = "cpu"
test_device = "cpu"
print(f"Using {device} device")
if Train_flag:
    writer = SummaryWriter(f"TrainLog/{paramfilename}/{datetime.datetime.today().strftime('%Y-%m-%d-%H_%M_%S')}")

"""file configs"""
path = 'Param/' + paramfilename
makefile(path)

"""loss function and optimizer"""
loss_fn = []
loss_fn1 = nn.MSELoss(reduction='sum')
loss_fn.append(loss_fn1)


"""Test model on measured data."""

test_dataset = DJIGetSingleTestData2D(device=device)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=20, shuffle=False)
print('Test data has been finished!')
Train = DJI2DTrainOnlyAmp(device=device)
# flag = torch.load(f"{path}/flag.pth")
# print(flag)
test_model = DJI2D()
test_model.to(device)
test_model.load_state_dict(torch.load(f"{path}/bs128_lr001_eph500_495_minvalidloss.pth"))

test_path = './Results'
print('Param has been loaded!')
Train.testing(test_dataloader, test_model, test_path, loss_fn)



# pattern_2D_out = Train.jili2pattern(dic['out']).detach().cpu().numpy()
# pattern_2D_label = Train.jili2pattern(dic['label']).detach().cpu().numpy()
#
# plt.subplot(121)
# plt.title('out')
# plt.pcolor(pattern_2D_out[0,0,:,:], cmap='jet')
# plt.subplot(122)
# plt.title('label')
# plt.pcolor(pattern_2D_label[0,0,:,:], cmap='jet')
# # plt.subplot(123)
# # plt.title('input')
# # plt.pcolor(dic['input'].detach().cpu().numpy()[0,0,:,:], cmap='jet')
# plt.show()