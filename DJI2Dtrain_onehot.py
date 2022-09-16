import sys
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from Utils.GetData import DJIGetData2D, DJIGetTestData2D
from Model.DJImodule import DJI2D
from Utils.makefile import makefile
from Utils.Time import Timer
from Utils.Train import DJI2DTrainOnlyAmp
def train():
    """config information"""
    cuda0 = "cuda:0" if torch.cuda.is_available() else "cpu"
    cuda1 = "cuda:1" if torch.cuda.is_available() else "cpu"
    cuda2 = "cuda:2" if torch.cuda.is_available() else "cpu"
    cuda3 = "cuda:3" if torch.cuda.is_available() else "cpu"

    """-----------------------------Hyper-parameters must be modified every time.-----------------------------"""
    batch_size = 128
    epoch_num = 500
    dataset_num = 50000
    train_valid_rate = 0.95
    warmup_step = 4000
    learning_rate = 0.001
    weight_decay = 1e-5

    """-----------------------------Filename of TrainLog and Param vary with parameters ---------------------------"""
    paramfilename = f'DJI2D_onehot CrossEntropy warmup{warmup_step} lr{learning_rate} dataset{dataset_num} bs{batch_size} epoch{epoch_num}'
    Train_flag = True
    time = Timer()
    device = cuda3
    test_device = "cpu"
    print(f"Using {device} device")
    if Train_flag:
        writer = SummaryWriter(f"TrainLog/{paramfilename}/{datetime.datetime.today().strftime('%Y-%m-%d-%H_%M_%S')}")

    """-----------------------------File configs-----------------------------"""
    path = 'Param/' + paramfilename
    makefile(path)

    """-----------------------------Load data-----------------------------"""
    # data shape (100000, 1, 121, 121)   (100000, 2, 3, 3)
    if Train_flag:
        dataset = DJIGetData2D(device=device)
        train_dataset, valid_dataset = random_split(
            dataset=dataset,
            lengths=[int(dataset_num * train_valid_rate), dataset_num - int(dataset_num * train_valid_rate)]
        )
        """数据集按照 9：1 划分训练集和验证集"""
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
        print('Data has been loaded')

        """model"""
        model = DJI2D()
        model.to(device=device)
        print(f'Model has been moved to {device}')

    """-----------------------------loss function and optimizer-----------------------------"""
    loss_fn = []
    loss_fn1 = nn.MSELoss()
    # loss_fn2 = nn.MSELoss(reduction='sum')
    loss_fn.append(loss_fn1)
    # loss_fn.append(loss_fn2)
    if Train_flag:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lambda1 = lambda epoch: min((epoch + 1) ** (-0.5), (epoch + 1) * warmup_step ** (-1.5))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        # scheduler = None
        print(f'optimizer start')

    """-----------------------------training-----------------------------"""
    Train = DJI2DTrainOnlyAmp(device=device)
    print('Start Training and Testing')
    if Train_flag:
        time.start()
        for epoch_index in range(epoch_num):
            if __name__ == "__main__":
                train_loss = Train.training(train_dataloader, model, loss_fn, batch_size, optimizer, scheduler)
                valid_loss = Train.validing(valid_dataloader, model, loss_fn, batch_size)
            writer.add_scalar('train/loss', train_loss, epoch_index + 1)
            writer.add_scalar('valid/loss', valid_loss, epoch_index + 1)

            if epoch_index % 5 == 0:
                try:
                    for name, parameters in model.named_parameters():
                        writer.add_histogram(f'param/{name}', parameters.detach(), epoch_index + 1)
                        writer.add_histogram(f'grad/{name}', parameters.grad.data, epoch_index + 1)
                except:
                    print(f"{epoch_index}'s histogram has something wrong.")
                    sys.exit(1)

            if epoch_index == 0:
                valid_loss_flag = valid_loss
                param_dict = model.state_dict()
            if valid_loss_flag > valid_loss:
                valid_loss_flag = valid_loss
                flag = epoch_index
                param_dict = model.state_dict()

            if (epoch_index % 5 == 0) & (epoch_index != 0):
                torch.save(model.state_dict(),
                           f"{path}/bs{batch_size}_"
                           f"lr{str(learning_rate).split('.')[-1]}_"
                           f"eph{epoch_num}"
                           f"_{epoch_index}_minvalidloss.pth")

            print(f'{epoch_index + 1} has finished, last {time.end():.2f}s.')
        print('Training and validating have been finished!')
        print(f'Total training and validating time is: {time.end():.2f} secs')

        """Save model parameters"""
        torch.save(param_dict, f"{path}/bs{batch_size}_"
                               f"lr{str(learning_rate).split('.')[-1]}"
                               f"_eph{epoch_num}"
                               f"_{flag}_totalminvalidloss.pth")
        torch.save(flag, f"{path}/flag.pth")
        print(f'Param of {flag} epoch have been saved!')

    """--------------------Test model on measured data.------------------------"""

    test_dataset = DJIGetTestData2D(device=device)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    print('Test data has been finished!')

    flag = torch.load(f"{path}/flag.pth")
    print(flag)
    test_model = DJI2D()
    test_model.to(device)
    test_model.eval()
    # flag = 173
    test_model.load_state_dict(torch.load(f"{path}/bs{batch_size}_"
                                          f"lr{str(learning_rate).split('.')[-1]}"
                                          f"_eph{epoch_num}"
                                          f"_{flag}_totalminvalidloss.pth"))
    # test_model.load_state_dict(torch.load(f"{path}/bs4096_lr0001_eph200_180_minvalidloss.pth"))

    test_path = './Results'
    print('Param has been loaded!')
    Train.testing(test_dataloader, test_model, test_path, loss_fn)

if __name__ == "__main__":
    train()