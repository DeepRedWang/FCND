from scipy.io import savemat, loadmat
import torch
import numpy as np
from matplotlib import pyplot as plt

class DJI2DTrainOnlyAmp():

    def __init__(self, device):

        self.mul = torch.tensor([0.2, 240], dtype=torch.float32, device=device)
        self.mul = self.mul.view(1, 2, 1, 1)
        self.add = torch.tensor([0.8, 120], dtype=torch.float32, device=device)
        self.add = self.mul.view(1, 2, 1, 1)

        self.weight = 0.1*torch.tensor(range(1,11), dtype=torch.float32, device=device).view(10,1)


    def mulCELoss(self, lossfn, out, X):
        loss = torch.tensor([0], dtype=torch.float32)
        for i in range(9):
            loss += lossfn(out[:,i,:].squeeze_(),X[:,i,:].squeeze_())
        return loss

    def training(self, train_dataloader, model, loss_fn, bs, optimizer, scheduler):
        model.train()
        train_loss = 0
        num_batches = len(train_dataloader)
        for batch_index, data in enumerate(train_dataloader):
            Y = data['echo']
            out = model(Y)

            X = data['target']

            # loss = self.mulCELoss(loss_fn[0], out, X)
            loss = loss_fn[0](out, X)
            # loss1 = loss_fn[1](out, X) / loss_fn[1](X, torch.zeros_like(X))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss = 10 * torch.log10(torch.tensor(train_loss) / torch.tensor(num_batches))

        return train_loss

    def validing(self, valid_dataloader, model, loss_fn, bs):
        model.eval()
        valid_loss = 0
        num_batches = len(valid_dataloader)
        with torch.no_grad():
            for batch_index, data in enumerate(valid_dataloader):
                Y = data['echo']
                out = model(Y)
                X = data['target']
                # loss = self.mulCELoss(loss_fn[0], out, X)
                loss = loss_fn[0](out, X)
                # loss = loss_fn[1](out, X) / loss_fn[1](X, torch.zeros_like(X))
                valid_loss += loss.item()

        valid_loss = 10 * torch.log10(torch.tensor(valid_loss) / torch.tensor(num_batches))

        return valid_loss

    def testing(self, test_dataloader, test_model, test_path, loss_fn):
        test_model.eval()
        test_loss = 0
        num_batches = len(test_dataloader)
        print(num_batches)
        with torch.no_grad():

            for batch_index, data in enumerate(test_dataloader):
                Y = data['echo']
                out = test_model(Y)
                print("The input and label is")
                X = data['target']
                X_weight = X@self.weight
                # savemat('./Results/network_excite_30.mat', {'network_excite_30':X_weight.squeeze_()})
                print(X_weight.squeeze_())

                print("The output is")
                out_weight = out@self.weight
                out_weight.squeeze_()
                savemat('./Results/network_excite_30.mat', {'network_excite_30': out_weight.numpy()})
                print(out_weight)
                # loss = self.mulCELoss(loss_fn[0], out, X)
                loss = loss_fn[0](out, X)
                test_loss += loss.item()/1
            # out = np.array(out.detach().cpu()).reshape((-1, 20, 20)).transpose(0, 2, 1)
            # savemat(f'{test_path}/' + f'results.mat', {'results': out})
        valid_loss = 10 * torch.log10(torch.tensor(test_loss) / torch.tensor(num_batches))
        print(valid_loss)

    def jili2pattern(self, jili):
        JI_amp = jili[:, 0, :, :].reshape((-1, 9))
        JI_phase = jili[:, 1, :, :].reshape((-1, 9))

        L_amp_JL = torch.sqrt(JI_amp * JI_amp / torch.sum(JI_amp * JI_amp, dim=-1, keepdim=True)).unsqueeze(dim=-2)
        L_phase_JL = JI_phase.unsqueeze(dim=-2)

        L_abs_theta = self.all_ata_abs_theta * L_amp_JL
        L_phase_theta = self.all_ata_phase_theta + L_phase_JL
        L_abs_phi = self.all_ata_abs_phi * L_amp_JL
        L_phase_phi = self.all_ata_phase_phi + L_phase_JL

        L_theta_vec = L_abs_theta * torch.exp(1j * L_phase_theta * torch.pi / 180)
        L_phi_vec = L_abs_phi * torch.exp(1j * L_phase_phi * torch.pi / 180)
        L_theta_vec_total = torch.sum(L_theta_vec, dim=-1)
        L_phi_vec_total = torch.sum(L_phi_vec, dim=-1)

        pattern = torch.abs(L_theta_vec_total * L_theta_vec_total) + torch.abs(L_phi_vec_total * L_phi_vec_total)
        # pattern = 10 * torch.log10(pattern)
        pattern_2D = pattern.reshape((-1, 121, 121)).permute(0, 2, 1)
        # pattern_2D = pattern_2D / torch.max(torch.max(torch.abs(pattern_2D), dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
        pattern_2D.unsqueeze_(dim=1)

        return pattern_2D



class DJI2DTrainCEAP():

    def __init__(self, device):

        self.mul = torch.tensor([0.2, 240], dtype=torch.float32, device=device)
        self.mul = self.mul.view(1, 2, 1, 1)
        self.add = torch.tensor([0.8, 120], dtype=torch.float32, device=device)
        self.add = self.mul.view(1, 2, 1, 1)

    def training(self, train_dataloader, model, loss_fn, bs, optimizer, scheduler):
        model.train()
        train_loss = 0
        num_batches = len(train_dataloader)
        for batch_index, data in enumerate(train_dataloader):
            Y = data['echo']
            out = model(Y)

            X = data['target']
            loss = loss_fn[0](out * self.mul + self.add, X)
            loss1 = loss_fn[1](out * self.mul + self.add, X) / loss_fn[0](X, torch.zeros_like(X))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss1.item()

        train_loss = 10 * torch.log10(torch.tensor(train_loss) / torch.tensor(num_batches))

        return train_loss

    def validing(self, valid_dataloader, model, loss_fn, bs):
        model.eval()
        valid_loss = 0
        num_batches = len(valid_dataloader)
        with torch.no_grad():
            for batch_index, data in enumerate(valid_dataloader):
                Y = data['echo']
                out = model(Y)
                X = data['target']
                loss = loss_fn[1](out * self.mul + self.add, X) / loss_fn[1](X, torch.zeros_like(X))
                valid_loss += loss.item()

        valid_loss = 10 * torch.log10(torch.tensor(valid_loss) / torch.tensor(num_batches))

        return valid_loss

    def testing(self, test_dataloader, test_model, test_path, loss_fn):
        test_model.eval()
        test_loss = 0
        num_batches = len(test_dataloader)
        print(num_batches)
        with torch.no_grad():

            for batch_index, data in enumerate(test_dataloader):
                Y = data['echo']
                out = test_model(Y)
                print("The input and label is")
                X = data['target']
                print(X)
                print("The output is")
                print(out * self.mul + self.add)
                loss = loss_fn[1](out * self.mul + self.add, X)
                test_loss += loss.item()/1
            # out = np.array(out.detach().cpu()).reshape((-1, 20, 20)).transpose(0, 2, 1)
            # savemat(f'{test_path}/' + f'results.mat', {'results': out})
        valid_loss = 10 * torch.log10(torch.tensor(test_loss) / torch.tensor(num_batches))
        print(valid_loss)

    def jili2pattern(self, jili):
        JI_amp = jili[:, 0, :, :].reshape((-1, 9))
        JI_phase = jili[:, 1, :, :].reshape((-1, 9))

        L_amp_JL = torch.sqrt(JI_amp * JI_amp / torch.sum(JI_amp * JI_amp, dim=-1, keepdim=True)).unsqueeze(dim=-2)
        L_phase_JL = JI_phase.unsqueeze(dim=-2)

        L_abs_theta = self.all_ata_abs_theta * L_amp_JL
        L_phase_theta = self.all_ata_phase_theta + L_phase_JL
        L_abs_phi = self.all_ata_abs_phi * L_amp_JL
        L_phase_phi = self.all_ata_phase_phi + L_phase_JL

        L_theta_vec = L_abs_theta * torch.exp(1j * L_phase_theta * torch.pi / 180)
        L_phi_vec = L_abs_phi * torch.exp(1j * L_phase_phi * torch.pi / 180)
        L_theta_vec_total = torch.sum(L_theta_vec, dim=-1)
        L_phi_vec_total = torch.sum(L_phi_vec, dim=-1)

        pattern = torch.abs(L_theta_vec_total * L_theta_vec_total) + torch.abs(L_phi_vec_total * L_phi_vec_total)
        # pattern = 10 * torch.log10(pattern)
        pattern_2D = pattern.reshape((-1, 121, 121)).permute(0, 2, 1)
        # pattern_2D = pattern_2D / torch.max(torch.max(torch.abs(pattern_2D), dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
        pattern_2D.unsqueeze_(dim=1)

        return pattern_2D



class DJI2DTrainAP():

    def __init__(self, device):

        self.mul = torch.tensor([0.2, 240], dtype=torch.float32, device=device)
        self.mul = self.mul.view(1, 2, 1, 1)
        self.add = torch.tensor([0.8, 120], dtype=torch.float32, device=device)
        self.add = self.mul.view(1, 2, 1, 1)

    def training(self, train_dataloader, model, loss_fn, bs, optimizer, scheduler):
        model.train()
        train_loss = 0
        num_batches = len(train_dataloader)
        for batch_index, data in enumerate(train_dataloader):
            Y = data['echo']
            out = model(Y)

            X = data['target']
            loss = loss_fn[0](out * self.mul + self.add, X)
            loss1 = loss_fn[0](out * self.mul + self.add, X) / loss_fn[0](X, torch.zeros_like(X))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss1.item()

        train_loss = 10 * torch.log10(torch.tensor(train_loss) / torch.tensor(num_batches))

        return train_loss

    def validing(self, valid_dataloader, model, loss_fn, bs):
        model.eval()
        valid_loss = 0
        num_batches = len(valid_dataloader)
        with torch.no_grad():
            for batch_index, data in enumerate(valid_dataloader):
                Y = data['echo']
                out = model(Y)
                X = data['target']
                loss = loss_fn[0](out * self.mul + self.add, X) / loss_fn[0](X, torch.zeros_like(X))
                valid_loss += loss.item()

        valid_loss = 10 * torch.log10(torch.tensor(valid_loss) / torch.tensor(num_batches))

        return valid_loss

    def testing(self, test_dataloader, test_model, test_path, loss_fn):
        test_model.eval()
        test_loss = 0
        num_batches = len(test_dataloader)
        print(num_batches)
        with torch.no_grad():

            for batch_index, data in enumerate(test_dataloader):
                Y = data['echo']
                out = test_model(Y)
                print("The input and label is")
                X = data['target']
                print(X)
                print("The output is")
                print(out * self.mul + self.add)
                loss = loss_fn[0](out * self.mul + self.add, X)
                test_loss += loss.item()/1
            # out = np.array(out.detach().cpu()).reshape((-1, 20, 20)).transpose(0, 2, 1)
            # savemat(f'{test_path}/' + f'results.mat', {'results': out})
        valid_loss = 10 * torch.log10(torch.tensor(test_loss) / torch.tensor(num_batches))
        print(valid_loss)

    def jili2pattern(self, jili):
        JI_amp = jili[:, 0, :, :].reshape((-1, 9))
        JI_phase = jili[:, 1, :, :].reshape((-1, 9))

        L_amp_JL = torch.sqrt(JI_amp * JI_amp / torch.sum(JI_amp * JI_amp, dim=-1, keepdim=True)).unsqueeze(dim=-2)
        L_phase_JL = JI_phase.unsqueeze(dim=-2)

        L_abs_theta = self.all_ata_abs_theta * L_amp_JL
        L_phase_theta = self.all_ata_phase_theta + L_phase_JL
        L_abs_phi = self.all_ata_abs_phi * L_amp_JL
        L_phase_phi = self.all_ata_phase_phi + L_phase_JL

        L_theta_vec = L_abs_theta * torch.exp(1j * L_phase_theta * torch.pi / 180)
        L_phi_vec = L_abs_phi * torch.exp(1j * L_phase_phi * torch.pi / 180)
        L_theta_vec_total = torch.sum(L_theta_vec, dim=-1)
        L_phi_vec_total = torch.sum(L_phi_vec, dim=-1)

        pattern = torch.abs(L_theta_vec_total * L_theta_vec_total) + torch.abs(L_phi_vec_total * L_phi_vec_total)
        # pattern = 10 * torch.log10(pattern)
        pattern_2D = pattern.reshape((-1, 121, 121)).permute(0, 2, 1)
        # pattern_2D = pattern_2D / torch.max(torch.max(torch.abs(pattern_2D), dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
        pattern_2D.unsqueeze_(dim=1)

        return pattern_2D


class DJI2DINFERAP():

    def __init__(self, device):
        # device = "cuda:0"
        Side_DJ_9 = loadmat('E:/WangJianyang/DJI/Data/training_validing/Side_DJ_9.mat')
        Side_DJ_9 = np.array(Side_DJ_9['Total_data'])
        Element_D_data = torch.tensor(Side_DJ_9[:, (0, 1, 3, 4, 5, 6), :], dtype=torch.float32, device=device)
        Element_D_data = Element_D_data.reshape((181, 360, 6, 9))
        Element_D_data = Element_D_data[30:151, 120:241, :, :].reshape(-1, 6, 9)
        # .reshape((-1, 181, 360)).permute(0, 2, 1)

        # self.weight = torch.tensor([1, 330], dtype=torch.float32, device=device).view(-1, 2, 1, 1)

        self.all_ata_abs_theta = torch.sqrt(Element_D_data[:, 2, :]).unsqueeze(dim=0)
        self.all_ata_phase_theta = Element_D_data[:, 3, :].unsqueeze(dim=0)
        self.all_ata_abs_phi = torch.sqrt(Element_D_data[:, 4, :]).unsqueeze(dim=0)
        self.all_ata_phase_phi = Element_D_data[:, 5, :].unsqueeze(dim=0)

        self.mul = torch.tensor([0.2, 240], dtype=torch.float32, device=device)
        self.mul = self.mul.view(1, 2, 1, 1)
        self.add = torch.tensor([0.8, 120], dtype=torch.float32, device=device)
        self.add = self.mul.view(1, 2, 1, 1)

    def training(self, train_dataloader, model, loss_fn, bs, optimizer, scheduler):
        model.train()
        train_loss = 0
        num_batches = len(train_dataloader)
        for batch_index, data in enumerate(train_dataloader):
            Y = data['echo']
            out = model(Y)
            X = data['target']
            loss = loss_fn[0](out * self.mul + self.add, X)
            loss1 = loss_fn[0](out * self.mul + self.add, X) / loss_fn[0](X, torch.zeros_like(X))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss1.item()

        train_loss = 10 * torch.log10(torch.tensor(train_loss) / torch.tensor(num_batches))

        return train_loss

    def validing(self, valid_dataloader, model, loss_fn, bs):
        model.eval()
        valid_loss = 0
        num_batches = len(valid_dataloader)
        with torch.no_grad():
            for batch_index, data in enumerate(valid_dataloader):
                Y = data['echo']
                out = model(Y)
                # amp = torch.sqrt(out[:, 0, :, :] * out[:, 0, :, :] + out[:, 1, :, :] * out[:, 1, :, :])
                # phase = torch.arctan(out[:, 1, :, :] / out[:, 0, :, :]) * torch.tensor(180) / torch.pi
                # out_amp_phase = torch.stack((amp, phase), dim=1)
                # out_pattern = self.jili2pattern(out_amp_phase)
                X = data['target']
                loss = loss_fn[0](out * self.mul + self.add, X) / loss_fn[0](X, torch.zeros_like(X))
                valid_loss += loss.item()

        valid_loss = 10 * torch.log10(torch.tensor(valid_loss) / torch.tensor(num_batches))

        return valid_loss

    def testing(self, test_dataloader, test_model, test_path, loss_fn):
        test_model.eval()
        test_loss = 0
        num_batches = len(test_dataloader)
        print(num_batches)
        for batch_index, data in enumerate(test_dataloader):
            X = data['target']
            print("The input and label is")
            print(X)

            # amp = torch.sqrt(out[:, 0, :, :] * out[:, 0, :, :] + out[:, 1, :, :] * out[:, 1, :, :])
            # phase = torch.arctan(out[:, 1, :, :] / out[:, 0, :, :]) * torch.tensor(180) / torch.pi
            # out_amp_phase = torch.stack((amp, phase), dim=1)
            # out_pattern = self.jili2pattern(out_amp_phase)
            Y = data['echo']
            out = test_model(Y)
            out = out * self.mul + self.add
            print("The output is")
            print(out)
            # print("The label is:\n")
            # print(X)
            # print(out.shape)
            # out = out/out.max(dim=-1, keepdim=True).values
            loss = loss_fn[0](out, X)

            test_loss += loss.item()/1
            # out = np.array(out.detach().cpu()).reshape((-1, 20, 20)).transpose(0, 2, 1)
            # savemat(f'{test_path}/' + f'results.mat', {'results': out})
        valid_loss = 10 * torch.log10(torch.tensor(test_loss) / torch.tensor(num_batches))
        print(valid_loss)
        return {'out':out, 'label':X, 'input':Y}

    def jili2pattern(self, jili):
        # JI_amp = jili[:, 0, :, :].reshape((-1, 9))
        # JI_phase = jili[:, 1, :, :].reshape((-1, 9))
        #
        # L_amp_JL = torch.sqrt(JI_amp * JI_amp / torch.sum(JI_amp * JI_amp, dim=-1, keepdim=True)).unsqueeze(dim=-2)
        # L_phase_JL = JI_phase.unsqueeze(dim=-2)
        L_amp_JL = jili[:,0,:,:].reshape((-1, 9))
        L_phase_JL = jili[:,1,:,:].reshape((-1, 9))

        L_abs_theta = self.all_ata_abs_theta * L_amp_JL
        L_phase_theta = self.all_ata_phase_theta + L_phase_JL
        L_abs_phi = self.all_ata_abs_phi * L_amp_JL
        L_phase_phi = self.all_ata_phase_phi + L_phase_JL

        L_theta_vec = L_abs_theta * torch.exp(1j * L_phase_theta * torch.pi / 180)
        L_phi_vec = L_abs_phi * torch.exp(1j * L_phase_phi * torch.pi / 180)
        L_theta_vec_total = torch.sum(L_theta_vec, dim=-1)
        L_phi_vec_total = torch.sum(L_phi_vec, dim=-1)

        pattern = torch.abs(L_theta_vec_total * L_theta_vec_total) + torch.abs(L_phi_vec_total * L_phi_vec_total)
        # pattern = 10 * torch.log10(pattern)
        pattern_2D = pattern.reshape((-1, 121, 121))
        # pattern_2D = pattern_2D / torch.max(torch.max(torch.abs(pattern_2D), dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
        pattern_2D.unsqueeze_(dim=1)

        return pattern_2D

class DJI2DTrain():

    def __init__(self, device):
        # device = "cuda:0"
        Side_DJ_9 = loadmat('E:/WangJianyang/DJI/Data/training_validing/Side_DJ_9.mat')
        Side_DJ_9 = np.array(Side_DJ_9['Total_data'])
        Element_D_data = torch.tensor(Side_DJ_9[:, (0, 1, 3, 4, 5, 6), :], dtype=torch.float32, device=device)
        Element_D_data = Element_D_data.reshape((181, 360, 6, 9))
        Element_D_data = Element_D_data[30:151, 140:261, :, :].reshape(-1, 6, 9)
        # .reshape((-1, 181, 360)).permute(0, 2, 1)

        self.weight = torch.tensor([1, 330], dtype=torch.float32, device=device).view(-1, 2, 1, 1)

        self.all_ata_abs_theta = torch.sqrt(Element_D_data[:, 2, :]).unsqueeze(dim=0)
        self.all_ata_phase_theta = Element_D_data[:, 3, :].unsqueeze(dim=0)
        self.all_ata_abs_phi = torch.sqrt(Element_D_data[:, 4, :]).unsqueeze(dim=0)
        self.all_ata_phase_phi = Element_D_data[:, 5, :].unsqueeze(dim=0)

    def training(self, train_dataloader, model, loss_fn, bs, optimizer, scheduler):
        """
        training
        """
        model.train()
        train_loss = 0
        num_batches = len(train_dataloader)
        print("num_batches:", num_batches)
        for batch_index, data in enumerate(train_dataloader):
            Y = data['echo']
            # X = data['target']
            out = model(Y)
            out_pattern = self.jili2pattern(out * self.weight)
            # X = X.softmax(dim=1)
            # loss = loss_fn[1](out, X)
            loss = loss_fn[0](out_pattern, Y)
            loss1 = loss_fn[0](out_pattern, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            train_loss += loss1.item()/bs

        train_loss = 10 * torch.log10(torch.tensor(train_loss) / torch.tensor(num_batches))

        return train_loss

    def validing(self, valid_dataloader, model, loss_fn, bs):
        """

        validing
        """
        model.eval()
        valid_loss = 0
        num_batches = len(valid_dataloader)
        with torch.no_grad():
            for batch_index, data in enumerate(valid_dataloader):
                Y = data['echo']
                out = model(Y)
                out_pattern = self.jili2pattern(out * self.weight)
                loss = loss_fn[0](out_pattern, Y)
                valid_loss += loss.item()/bs

        valid_loss = 10 * torch.log10(torch.tensor(valid_loss) / torch.tensor(num_batches))

        return valid_loss

    def testing(self, test_dataloader, test_model, test_path, loss_fn):
        """

        testing
        """
        test_model.eval()
        test_loss = 0
        num_batches = len(test_dataloader)
        print(num_batches)
        for batch_index, data in enumerate(test_dataloader):
            Y = data['echo']
            out = test_model(Y)
            print("The input and label is")
            print(Y)
            print("The output is")
            amp = torch.sqrt(out[:, 0, :, :] * out[:, 0, :, :] + out[:, 1, :, :] * out[:, 1, :, :])
            phase = torch.arctan(out[:, 1, :, :] / out[:, 0, :, :]) * torch.tensor(180) / torch.pi
            out_amp_phase = torch.stack((amp, phase), dim=1)
            out_pattern = self.jili2pattern(out_amp_phase)
            print(out_pattern)
            plt.pcolor(out_pattern.cpu().detach().numpy()[0, 0, :, :], cmap='jet')
            # plt.pcolor(out_pattern.cpu().detach().numpy()[0, 0, :, :], cmap='jet')
            plt.show()
            # print("The label is:\n")
            # print(X)
            # print(out.shape)
            # out = out/out.max(dim=-1, keepdim=True).values
            loss = loss_fn[0](out_pattern, Y)

            test_loss += loss.item()/1
            # out = np.array(out.detach().cpu()).reshape((-1, 20, 20)).transpose(0, 2, 1)
            # savemat(f'{test_path}/' + f'results.mat', {'results': out})
        valid_loss = 10 * torch.log10(torch.tensor(test_loss) / torch.tensor(num_batches))
        print(valid_loss)

    def jili2pattern(self, jili):
        """from jili to pattern"""
        JI_amp = jili[:, 0, :, :].reshape((-1, 9))
        JI_phase = jili[:, 1, :, :].reshape((-1, 9))

        L_amp_JL = torch.sqrt(JI_amp * JI_amp / torch.sum(JI_amp * JI_amp, dim=-1, keepdim=True)).unsqueeze(dim=-2)
        L_phase_JL = JI_phase.unsqueeze(dim=-2)

        L_abs_theta = self.all_ata_abs_theta * L_amp_JL
        L_phase_theta = self.all_ata_phase_theta + L_phase_JL
        L_abs_phi = self.all_ata_abs_phi * L_amp_JL
        L_phase_phi = self.all_ata_phase_phi + L_phase_JL

        L_theta_vec = L_abs_theta * torch.exp(1j * L_phase_theta * torch.pi / 180)
        L_phi_vec = L_abs_phi * torch.exp(1j * L_phase_phi * torch.pi / 180)
        L_theta_vec_total = torch.sum(L_theta_vec, dim=-1)
        L_phi_vec_total = torch.sum(L_phi_vec, dim=-1)

        pattern = torch.abs(L_theta_vec_total * L_theta_vec_total) + torch.abs(L_phi_vec_total * L_phi_vec_total)
        pattern_dB = 10 * torch.log10(pattern)
        pattern_2D = pattern_dB.reshape((-1, 121, 121)).permute(0, 2, 1)
        pattern_2D = pattern_2D / torch.max(torch.max(torch.abs(pattern_2D), dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
        pattern_2D.unsqueeze_(dim=1)

        return pattern_2D






# class DJITrain():
#
#     def __init__(self):
#         pass
#     def training(self, train_dataloader, model, loss_fn, bs, optimizer, scheduler):
#         model.train()
#         train_loss = 0
#         num_batches = len(train_dataloader)
#         for batch_index, data in enumerate(train_dataloader):
#             Y = data['echo']
#             X = data['target']
#             out = model(Y)
#
#             # X = X.softmax(dim=1)
#             # loss = loss_fn[1](out, X)
#             loss = loss_fn[0](out, X)
#             loss1 = loss_fn[0](out, X)/loss_fn[0](X, torch.zeros_like(X))
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             # scheduler.step()
#             train_loss += loss1.item()/bs
#
#         train_loss = 10 * torch.log10(torch.tensor(train_loss) / torch.tensor(num_batches))
#
#         return train_loss
#
#     def validing(self, valid_dataloader, model, loss_fn, bs):
#         model.eval()
#         valid_loss = 0
#         num_batches = len(valid_dataloader)
#         with torch.no_grad():
#             for batch_index, data in enumerate(valid_dataloader):
#                 Y = data['echo']
#                 X = data['target']
#                 out = model(Y)
#                 loss = loss_fn[0](out, X)/loss_fn[0](X, torch.zeros_like(X))
#                 valid_loss += loss.item()/bs
#
#         valid_loss = 10 * torch.log10(torch.tensor(valid_loss) / torch.tensor(num_batches))
#
#         return valid_loss
#
#     def testing(self, test_dataloader, test_model, test_path, loss_fn):
#         test_model.eval()
#         test_loss = 0
#         num_batches = len(test_dataloader)
#         print(num_batches)
#         for batch_index, data in enumerate(test_dataloader):
#             Y = data['echo']
#             X = data['target']
#             out = test_model(Y)
#             print("The output is")
#             print(out)
#             print("The label is")
#             print(X)
#             # print("The label is:\n")
#             # print(X)
#             # print(out.shape)
#             out = out/out.max(dim=-1, keepdim=True).values
#             loss = loss_fn[0](out, X)/loss_fn[0](torch.zeros_like(X), X)
#
#             test_loss += loss.item()/1
#             out = np.array(out.detach().cpu()).reshape((-1, 20, 20)).transpose(0, 2, 1)
#             savemat(f'{test_path}/' + f'results.mat', {'results': out})
#         valid_loss = 10 * torch.log10(torch.tensor(test_loss) / torch.tensor(num_batches))
#         print(valid_loss)
