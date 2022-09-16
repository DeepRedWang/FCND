import torch
import torch.nn as nn
import torch.nn.functional as F

class startmodule(nn.Module):

    def __init__(self, in_chan, mid_chan):
        super(startmodule, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=mid_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_chan),
            nn.GELU()
        )
    def forward(self, x):
        return self.resblock(x)

class endmodule(nn.Module):

    def __init__(self, in_chan, mid_chan):
        super(endmodule, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=mid_chan, kernel_size=3, padding=1),
        )
    def forward(self, x):
        return self.resblock(x)

class slmodule(nn.Module):

    def __init__(self, in_chan, mid_chan):
        super(slmodule, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=mid_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_chan),
            nn.GELU(),
            nn.Conv2d(in_channels=mid_chan, out_channels=mid_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_chan),
        )
        self.activate = nn.GELU()

    def forward(self, x):
        return self.activate(self.resblock(x) + x)

class Basicmodule(nn.Module):

    def __init__(self, in_chan, mid_chan):
        super(Basicmodule, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=mid_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_chan),
            nn.GELU(),
            nn.Conv2d(in_channels=mid_chan, out_channels=mid_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_chan),
        )
        self.activate = nn.GELU()

    def forward(self, x):
        return self.activate(self.resblock(x) + x)

class Skipmodule(nn.Module):
    def __init__(self, in_chan, mid_chan):
        super(Skipmodule, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=mid_chan, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(mid_chan),
            nn.GELU(),
            nn.Conv2d(in_channels=mid_chan, out_channels=mid_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_chan),
        )
        self.activate = nn.GELU()
        self.skip = nn.Conv2d(in_channels=in_chan, out_channels=mid_chan, kernel_size=3, padding=1, stride=2)
    def forward(self, x):
        return self.activate(self.resblock(x) + self.skip(x))

class DJI(nn.Module):
    def __init__(self):
        super(DJI, self).__init__()
        self.resgroup = nn.Sequential(
            startmodule(1,64),
            Basicmodule(64, 64),
            Basicmodule(64, 64),
            Skipmodule(64, 128),
            Basicmodule(128, 128),
            Basicmodule(128, 128),
            Skipmodule(128, 256),
            Basicmodule(256, 256),
            Basicmodule(256, 256),
            endmodule(256, 2)
        )
        # self.Linear(64*18, 18)
    def forward(self, x):
        return self.resgroup(x)





class start2Dmodule(nn.Module):

    def __init__(self, in_chan, mid_chan):
        super(start2Dmodule, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=mid_chan, kernel_size=7, padding=3, stride=3),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU()
        )
    def forward(self, x):
        return self.resblock(x)

class end2Dmodule(nn.Module):

    def __init__(self, in_chan, mid_chan):
        super(end2Dmodule, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=mid_chan, kernel_size=2, stride=2)
            # nn.BatchNorm2d(mid_chan),
            # nn.Sigmoid()
        )
    def forward(self, x):
        return self.resblock(x)

class end2DAPmodule(nn.Module):

    def __init__(self, in_chan, mid_chan):
        super(end2DAPmodule, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=mid_chan, kernel_size=3, stride=1)

            # nn.BatchNorm2d(mid_chan),
            # nn.Sigmoid()
        )
    def forward(self, x):
        return self.resblock(x)

class Basic2Dmodule(nn.Module):

    def __init__(self, in_chan, mid_chan):
        super(Basic2Dmodule, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=mid_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_chan, out_channels=mid_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_chan),
        )
        self.activate = nn.ReLU()

    def forward(self, x):
        return self.activate(self.resblock(x) + x)

class Skip2Dmodule(nn.Module):
    def __init__(self, in_chan, mid_chan):
        super(Skip2Dmodule, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=mid_chan, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU()
            # nn.Conv2d(in_channels=mid_chan, out_channels=mid_chan, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_chan),
        )
        # self.activate = nn.GELU()
        # self.skip = nn.Conv2d(in_channels=in_chan, out_channels=mid_chan, kernel_size=3, padding=1, stride=2)
    def forward(self, x):
        return self.resblock(x)


class DJI2D(nn.Module):

    def __init__(self):
        super(DJI2D, self).__init__()
        self.resgroup = nn.Sequential(
            start2Dmodule(1, 32),
            Basic2Dmodule(32, 32),
            Basic2Dmodule(32, 32),

            Skip2Dmodule(32, 32),

            Basic2Dmodule(32, 32),
            Basic2Dmodule(32, 32),
            Skip2Dmodule(32, 64),
            Basic2Dmodule(64, 64),
            Basic2Dmodule(64, 64),
            Skip2Dmodule(64, 128),
            Basic2Dmodule(128, 128),
            Basic2Dmodule(128, 128)
        )
        self.Linear = nn.Linear(4608, 90)
        self.softmax = nn.Softmax(dim=-1)

        # self.Linear(64*18, 18)

    def forward(self, x):
        y1 = self.resgroup(x)
        y1 = y1.view(-1,1,4608)
        # out = self.Linear(y1)
        # out = out.view(-1, 9, 10)
        # out = self.softmax(out)
        return y1

# a = DJI2D()
# input_value = torch.rand(100, 1, 121, 121)
# print(a(input_value).shape)
# input_value = torch.rand(100, 121, 121)
# # print(k.shape)
# # # print(input.shape[1])
# # # input = input.resize((input.shape[0],1,input.shape[1],1))
# out = a(input_value)
# print(out.shape)

# input_value = torch.tensor([[[1.,2],[2.,4]],[[3.,6],[4.,9]]])
# out = nn.Softmax(dim=-1)(input_value)
# print(input_value)
# print(out)