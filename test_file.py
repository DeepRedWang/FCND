import torch
import torch.nn as nn

# model = nn.Linear(10,20)
# input = torch.rand((5,10))
# loss_fn = torch.nn.MSELoss()
# warmup_step = 50
# optimizer = torch.optim.Adam(model.parameters(), lr=0.125)
# lambda1 = lambda epoch: min(epoch**(-0.5), epoch*warmup_step**(-1.5))+1e-8
# lambda2 = lambda epoch: 0.95 ** epoch
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
# for i in range(100):
#     input = torch.rand((5, 10))
#     output = torch.rand((5,20))
#     predict = model(input)
#     loss = loss_fn(predict, output)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     scheduler.step()
#     now_lr = scheduler.get_last_lr()
#     print(now_lr)
# from torch.nn import TransformerEncoderLayer, TransformerEncoder
# import torch.nn as nn
# # model = TransformerEncoderLayer(512, 8)
# encoder_layer = TransformerEncoderLayer(d_model=64, dim_feedforward=256, nhead=8, batch_first=True)
# # model = TransformerEncoder(encoder_layer, 4)
# data = torch.ones((32, 256, 64))
# out = encoder_layer(data)
# print(out.shape)
#
# import torch
# a = torch.log(torch.tensor(10))
# print(a)
#
# loss_fn1 = nn.MSELoss(reduction='sum')
# nn.KLDivLoss
# a = torch.tensor([[2., 2,3], [2,3,4]])
# a = torch.zeros((3, 4, 5))
# b = torch.tensor([[2., 3,4], [1,2,3]])
# b = torch.ones((3, 4, 5))
# loss = loss_fn1(a, b)
# print(loss/3)

a = float(input("time rate\n"))
k = float(input("accurate rate\n"))
S = 1/((1-a)+a/k)
print(f"answer is {S}")
