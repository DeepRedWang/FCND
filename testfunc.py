from scipy.io import loadmat
import numpy as np
import torch
from matplotlib import pyplot as plt
import h5py
"""
device = "cuda:0"
Side_DJ_9 = loadmat('E:/WangJianyang/DJI/Data/training_validing/Side_DJ_9.mat')
Side_DJ_9 = np.array(Side_DJ_9['Total_data'])
Element_D_data = torch.tensor(Side_DJ_9[:, (0, 1, 3, 4, 5, 6), :], dtype=torch.float32, device=device)

JI_amp = torch.ones((3, 3), device=device)
JI_phase = torch.zeros((3, 3), device=device)
jili = torch.stack((JI_amp, JI_phase), dim=0)

JI_amp2 = torch.ones((3, 3), device=device)
JI_phase2 = 55 * torch.tensor([1, 2, 3, 4, 1, 2, 3, 4, 1], device=device).reshape((3, 3))
jili2 = torch.stack((JI_amp2, JI_phase2), dim=0)
jili = torch.stack((jili, jili2), dim=0)
print(jili.shape)



JI_amp = jili[:, 0, :, :].reshape((-1, 9))
JI_phase = jili[:, 1, :, :].reshape((-1, 9))

L_amp_JL = torch.sqrt(JI_amp*JI_amp/torch.sum(JI_amp*JI_amp, dim=-1, keepdim=True)).unsqueeze(dim=-2)
print(L_amp_JL.shape)
L_phase_JL = JI_phase.unsqueeze(dim=-2)

all_ata_abs_theta = torch.sqrt(Element_D_data[:, 2, :]).unsqueeze(dim=0)
all_ata_phase_theta = Element_D_data[:, 3, :].unsqueeze(dim=0)
all_ata_abs_phi = torch.sqrt(Element_D_data[:, 4, :]).unsqueeze(dim=0)
all_ata_phase_phi = Element_D_data[:, 5, :].unsqueeze(dim=0)


L_abs_theta = all_ata_abs_theta * L_amp_JL
L_phase_theta = all_ata_phase_theta + L_phase_JL
L_abs_phi = all_ata_abs_phi * L_amp_JL
L_phase_phi = all_ata_phase_phi + L_phase_JL

L_theta_vec = L_abs_theta * torch.exp(1j * L_phase_theta * torch.pi / 180)
L_phi_vec = L_abs_phi * torch.exp(1j * L_phase_phi * torch.pi / 180)

L_theta_vec_total = torch.sum(L_theta_vec, dim=-1)
L_phi_vec_total = torch.sum(L_phi_vec, dim=-1)

# pattern = torch.abs(L_theta_vec_total) * torch.abs(L_theta_vec_total) + L_phi_vec_total * L_phi_vec_total
pattern = torch.abs(L_theta_vec_total * L_theta_vec_total) + torch.abs(L_phi_vec_total * L_phi_vec_total)

pattern_dB = 10 * torch.log10(pattern)
# print(L_abs_theta.shape)
# print(L_phase_theta.shape)
# print(L_theta_vec.dtype)
# print(L_theta_vec_total.shape)
#
print(pattern_dB[1, 0:5])
pattern_2D = pattern_dB.reshape((-1, 181, 360)).permute(0, 2, 1)
print("normal shape is", torch.max(torch.max(pattern_2D[:, 140:261, 30:151], dim=-1, keepdim=True).values, dim=-2, keepdim=True).values.shape)
pattern_2D = pattern_2D[:, 140:261, 30:151] / torch.max(torch.max(torch.abs(pattern_2D[:, 140:261, 30:151]), dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
# print(pattern_2D[1, 140:261, 30:151])

print(pattern_2D.shape)
pattern_2D.unsqueeze_(dim=1)
print(pattern_2D.shape)
print(pattern_2D[0, 0, 0:5, 0:5])
plt.pcolor(pattern_2D[1, 0, :, :].cpu(), cmap='jet')
plt.show()
"""

#
# Y = h5py.File(f'E:/WangJianyang/DJI/Data/Data-3D/log_abs_total_3D_all.mat')
# Y = np.transpose(Y['log_abs_total_3D_all'])
# Y = torch.tensor(Y, dtype=torch.float32)
# Y.unsqueeze_(dim=1)
# Y = Y / torch.max(torch.max(torch.abs(Y), dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
# print(Y.shape)
# print(Y[0, 0, :, :])



# result = [1,2]
# def a(result):
#     result.pop()
#
# a(result)
# print(result)
import torch
import torch.nn as nn
# Y = h5py.File(f'./data/5G_direct_Coe.mat')
# Y = np.transpose(Y['direct_Coe'])
# direction = torch.tensor(Y, dtype=torch.float32)
# print(direction.shape)
# direction.unsqueeze_(dim=1)
#
# Y = loadmat(f'./data/5G_excite_OneHot.mat')
# Y = np.array(Y['excite_OneHot'])
# direction = torch.tensor(Y, dtype=torch.float32)
# print(direction.shape)
# direction.unsqueeze_(dim=1)

a = torch.rand((2,3))
b = torch.rand((2))
p = nn.CrossEntropyLoss()
g = p(a,b)
print(a.shape)





# device = "cuda:0"
# Side_DJ_9 = loadmat('E:/WangJianyang/DJI/Data/training_validing/Side_DJ_9.mat')
# Side_DJ_9 = np.array(Side_DJ_9['Total_data'])
# Element_D_data = torch.tensor(Side_DJ_9[:, (0, 1, 3, 4, 5, 6), :], dtype=torch.float32, device=device)
# Element_D_data = Element_D_data.reshape((181, 360, 6, 9))
# Element_D_data = Element_D_data[30:151, 140:261, :, :].reshape(-1, 6, 9)
#
#
#
#
# JI_amp = torch.ones((3, 3), device=device)
# JI_phase = torch.zeros((3, 3), device=device)
# jili = torch.stack((JI_amp, JI_phase), dim=0)
#
# JI_amp2 = torch.ones((3, 3), device=device)
# JI_phase2 = 55 * torch.tensor([1, 2, 3, 4, 1, 2, 3, 4, 1], device=device).reshape((3, 3))
# jili2 = torch.stack((JI_amp2, JI_phase2), dim=0)
# jili = torch.stack((jili, jili2), dim=0)
# print(jili.shape)
#
#
#
# JI_amp = jili[:, 0, :, :].reshape((-1, 9))
# JI_phase = jili[:, 1, :, :].reshape((-1, 9))
#
# L_amp_JL = torch.sqrt(JI_amp*JI_amp/torch.sum(JI_amp*JI_amp, dim=-1, keepdim=True)).unsqueeze(dim=-2)
# print(L_amp_JL.shape)
# L_phase_JL = JI_phase.unsqueeze(dim=-2)
#
# all_ata_abs_theta = torch.sqrt(Element_D_data[:, 2, :]).unsqueeze(dim=0)
# all_ata_phase_theta = Element_D_data[:, 3, :].unsqueeze(dim=0)
# all_ata_abs_phi = torch.sqrt(Element_D_data[:, 4, :]).unsqueeze(dim=0)
# all_ata_phase_phi = Element_D_data[:, 5, :].unsqueeze(dim=0)
#
#
# L_abs_theta = all_ata_abs_theta * L_amp_JL
# L_phase_theta = all_ata_phase_theta + L_phase_JL
# L_abs_phi = all_ata_abs_phi * L_amp_JL
# L_phase_phi = all_ata_phase_phi + L_phase_JL
#
# L_theta_vec = L_abs_theta * torch.exp(1j * L_phase_theta * torch.pi / 180)
# L_phi_vec = L_abs_phi * torch.exp(1j * L_phase_phi * torch.pi / 180)
#
# L_theta_vec_total = torch.sum(L_theta_vec, dim=-1)
# L_phi_vec_total = torch.sum(L_phi_vec, dim=-1)
#
# # pattern = torch.abs(L_theta_vec_total) * torch.abs(L_theta_vec_total) + L_phi_vec_total * L_phi_vec_total
# pattern = torch.abs(L_theta_vec_total * L_theta_vec_total) + torch.abs(L_phi_vec_total * L_phi_vec_total)
#
# pattern_dB = 10 * torch.log10(pattern)
# # print(L_abs_theta.shape)
# # print(L_phase_theta.shape)
# # print(L_theta_vec.dtype)
# # print(L_theta_vec_total.shape)
# #
# # print(pattern_dB[1, 0:5])
# pattern_2D = pattern_dB.reshape((-1, 121, 121)).permute(0, 2, 1)
# print("normal shape is", torch.max(torch.max(pattern_2D, dim=-1, keepdim=True).values, dim=-2, keepdim=True).values.shape)
# pattern_2D = pattern_2D / torch.max(torch.max(torch.abs(pattern_2D), dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
# # print(pattern_2D[1, 140:261, 30:151])
#
# print(pattern_2D.shape)
# pattern_2D.unsqueeze_(dim=1)
# print(pattern_2D.shape)
# print(pattern_2D[0, 0, 0:5, 0:5])
# plt.pcolor(pattern_2D[1, 0, :, :].cpu(), cmap='jet')
# plt.show()







"""----------------------------numpy----------------------------"""

# from scipy.io import loadmat
# import numpy as np
# from matplotlib import pyplot as plt
#
# path = 'E:/WangJianyang/DJI/Data/training_validing/Side_DJ_9.mat'
# Side_DJ_9 = loadmat(path)
# Side_DJ_9 = np.array(Side_DJ_9['Total_data'])
# Element_D_data = np.array(Side_DJ_9[32400:32760, (0, 1, 3, 4, 5, 6), :], dtype=np.float32)
#
# JI_amp = np.ones((3, 3))
# JI_phase = np.zeros((3, 3))
# # jili =
# JI_amp = JI_amp.reshape((9))
# JI_phase = JI_phase.reshape((9))
#
# JI_phase = 55 * np.array([1, 2, 3, 4, 1, 2, 3, 4, 1])
# L_amp_JL = np.sqrt(JI_amp*JI_amp/np.sum(JI_amp*JI_amp))
# L_phase_JL = JI_phase
#
# all_ata_abs_theta = np.sqrt(Element_D_data[:, 2, :])
# all_ata_phase_theta = Element_D_data[:, 3, :]
# all_ata_abs_phi = np.sqrt(Element_D_data[:, 4, :])
# all_ata_phase_phi = Element_D_data[:, 5, :]
#
# L_abs_theta = all_ata_abs_theta * L_amp_JL
# L_phase_theta = all_ata_phase_theta + L_phase_JL
# L_abs_phi = all_ata_abs_phi * L_amp_JL
# L_phase_phi = all_ata_phase_phi + L_phase_JL
#
# L_theta_vec = L_abs_theta * np.exp(1j * L_phase_theta * np.pi / 180)
# L_phi_vec = L_abs_phi * np.exp(1j * L_phase_phi * np.pi / 180)
#
# L_theta_vec_total = np.sum(L_theta_vec, axis=1)
# L_phi_vec_total = np.sum(L_phi_vec, axis=1)
#
# pattern = np.abs(L_theta_vec_total * L_theta_vec_total) + np.abs(L_phi_vec_total * L_phi_vec_total)
# pattern_dB = 10 * np.log10(pattern)
#
#
# print(np.max(pattern_dB))
# plt.plot(pattern_dB)
# plt.show()