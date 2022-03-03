import torch
import torch.nn as nn
import math
from util import *
from keras.utils.np_utils import to_categorical

def onehot_labelling(int_labels, num_classes):
    categorical_labels = to_categorical(int_labels, num_classes=num_classes)
    return categorical_labels


class Mask_PseudoSCLLoss(nn.Module):
    def __init__(self, batch_size, temperature, device, num_labels_OOD):
        super(Mask_PseudoSCLLoss, self).__init__()
        self.num_labels_OOD = num_labels_OOD
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        #self.mask = self.mask_correlated_samples(batch_size)
        #self.criterion = nn.CrossEntropyLoss(reduction="sum")


    def mask_temprature_samples(self, batch_size, c_i, c_j, threshold):
        #c = torch.cat((c_i, c_j), dim=0)
        t_mask_i = torch.matmul(c_i, c_i.T)
        t_mask_j = torch.matmul(c_j, c_j.T)
        t_mask_ij = torch.matmul(c_i, c_j.T)

        t_mask_i[t_mask_i < threshold] = self.temperature
        t_mask_j[t_mask_j < threshold] = self.temperature
        t_mask_ij[t_mask_ij < threshold] = self.temperature

        t_mask_i[t_mask_i >= threshold] = 1
        t_mask_j[t_mask_j >= threshold] = 1
        t_mask_ij[t_mask_ij >= threshold] = 1

        t_mask_i = t_mask_i.fill_diagonal_(self.temperature)
        t_mask_j = t_mask_j.fill_diagonal_(self.temperature)
        t_mask_ij = t_mask_ij.fill_diagonal_(self.temperature)
        #logger.debug(t_mask)

        return t_mask_i, t_mask_j, t_mask_ij

    def getPseudoLabel(self, c_i, c_j, threshold):
        label_mask_i = torch.matmul(c_i, c_i.T)
        label_mask_j = torch.matmul(c_j, c_j.T)
        label_mask_ij = torch.matmul(c_i, c_j.T)

        label_mask_i[label_mask_i < threshold] = 0
        label_mask_j[label_mask_j < threshold] = 0
        label_mask_ij[label_mask_ij < threshold] = 0

        label_mask_i[label_mask_i >= threshold] = 1
        label_mask_j[label_mask_j >= threshold] = 1
        label_mask_ij[label_mask_ij >= threshold] = 1

        label_mask_i = label_mask_i.fill_diagonal_(1)
        label_mask_j = label_mask_j.fill_diagonal_(1)
        label_mask_ij = label_mask_ij.fill_diagonal_(1)

        return label_mask_i, label_mask_j, label_mask_ij


    def pair_cosine_similarity(self, x, x_adv, eps=1e-8):
        n = x.norm(p=2, dim=1, keepdim=True)
        n_adv = x_adv.norm(p=2, dim=1, keepdim=True)
        return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_adv @ x_adv.t()) / (n_adv * n_adv.t()).clamp(min=eps), (
                    x @ x_adv.t()) / (n * n_adv.t()).clamp(min=eps)

    def nt_xent(self, x, x_adv, label_mask_i, label_mask_j, label_mask_ij, t_mask_i, t_mask_j, t_mask_ij, cuda=True):
        '''
        t_mask_i, t_mask_j, t_mask_ij = self.mask_temprature_samples(self.batch_size, c_i, c_j)
        t_mask_i = t_mask_i.detach()
        t_mask_j = t_mask_j.detach()
        t_mask_ij = t_mask_ij.detach()
        '''
        t_mask_i = t_mask_i.detach()
        t_mask_j = t_mask_j.detach()
        t_mask_ij = t_mask_ij.detach()

        label_mask_i = label_mask_i.detach()
        label_mask_j = label_mask_j.detach()
        label_mask_ij = label_mask_ij.detach()

        x, x_adv, x_c = self.pair_cosine_similarity(x, x_adv)
        x = torch.exp(x / t_mask_i)
        x_adv = torch.exp(x_adv / t_mask_j)
        x_c = torch.exp(x_c / t_mask_ij)
        mask_count_i = label_mask_i.sum(1)
        mask_count_j = label_mask_j.sum(1)
        mask_count_ij = label_mask_ij.sum(1)
        mask_count_ji = label_mask_ij.sum(0)

        mask_count_1 = mask_count_i + mask_count_ij - 1
        mask_count_2 = mask_count_j + mask_count_ji - 1

        #mask_count = torch.cat((mask_count_i, mask_count_j), dim=0)
        #mask_count_2 = torch.cat((mask_count_ij, mask_count_ji), dim=0)
        #mask_count = mask_count + mask_count_2

        mask_reverse = (~(label_mask_ij.bool())).long()
        if cuda:
            dis = (x * (label_mask_i - torch.eye(x.size(0)).long().cuda()) + x_c * label_mask_ij) / (
                        x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / self.temperature))) + mask_reverse
            dis_adv = (x_adv * (label_mask_j - torch.eye(x.size(0)).long().cuda()) + x_c.T * label_mask_ij.T) / (
                        x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / self.temperature))) + mask_reverse.T
        else:
            dis = (x * (label_mask_i - torch.eye(x.size(0)).long()) + x_c * label_mask_ij) / (
                        x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / self.temperature))) + mask_reverse
            dis_adv = (x_adv * (label_mask_j - torch.eye(x.size(0)).long()) + x_c.T * label_mask_ij.T) / (
                        x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / self.temperature))) + mask_reverse.T
        loss = (torch.log(dis).sum(1)) / mask_count_1 + (torch.log(dis_adv).sum(1)) / mask_count_2
        #loss = (torch.log(dis).sum(1)) / mask_count
        # loss = dis.sum(1) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + dis_adv.sum(1) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t)))
        return -loss.mean()
        # return -torch.log(loss).mean()

    def forward(self, z_i, z_j, c_i, c_j):
        self.batch_size = len(z_i)
        t_mask_i, t_mask_j, t_mask_ij = self.mask_temprature_samples(self.batch_size, c_i, c_j, threshold=0.5)
        label_mask_i, label_mask_j, label_mask_ij = self.getPseudoLabel(c_i, c_j, threshold=0.5)
        sup_cont_loss = self.nt_xent(z_i, z_j, label_mask_i, label_mask_j, label_mask_ij, t_mask_i, t_mask_j, t_mask_ij, cuda=True)
        '''
        if len(z_i) > self.batch_size/2:
            label_mask = self.mask_correlated_samples(preds_i, preds_j)
            sup_cont_loss = self.nt_xent(z_i, z_j, label_mask, cuda=True)
        else:
            return 0
        '''


        '''
        preds_i, preds_j, z_i, z_j = self.getPseudoLabel(z_i, z_j, c_i, c_j, threshold=0)
        label_mask = self.mask_correlated_samples(preds_i, preds_j)
        sup_cont_loss = self.nt_xent(z_i, z_j, c_i, c_j, label_mask, cuda=True)
        '''

        return sup_cont_loss

class PseudoSCLLoss(nn.Module):
    def __init__(self, batch_size, temperature, device, num_labels_OOD):
        super(PseudoSCLLoss, self).__init__()
        self.num_labels_OOD = num_labels_OOD
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        #self.mask = self.mask_correlated_samples(batch_size)
        #self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_temprature_samples(self, batch_size, c_i, c_j):
        #c = torch.cat((c_i, c_j), dim=0)
        t_mask_i = torch.matmul(c_i, c_i.T)
        t_mask_j = torch.matmul(c_j, c_j.T)
        t_mask_ij = torch.matmul(c_i, c_j.T)

        t_mask_i[t_mask_i < self.temperature] = self.temperature
        t_mask_j[t_mask_j < self.temperature] = self.temperature

        t_mask_ij[t_mask_ij < self.temperature] = self.temperature
        t_mask_ij = t_mask_ij.fill_diagonal_(self.temperature)
        #logger.debug(t_mask)

        return t_mask_i, t_mask_j, t_mask_ij

    def getPseudoLabel(self, c_i, c_j, threshold):
        label_mask_i = torch.matmul(c_i, c_i.T)
        label_mask_j = torch.matmul(c_j, c_j.T)
        label_mask_ij = torch.matmul(c_i, c_j.T)

        label_mask_i[label_mask_i < threshold] = 0
        label_mask_j[label_mask_j < threshold] = 0
        label_mask_ij[label_mask_ij < threshold] = 0

        label_mask_i[label_mask_i >= threshold] = 1
        label_mask_j[label_mask_j >= threshold] = 1
        label_mask_ij[label_mask_ij >= threshold] = 1

        label_mask_i = label_mask_i.fill_diagonal_(1)
        label_mask_j = label_mask_j.fill_diagonal_(1)
        label_mask_ij = label_mask_ij.fill_diagonal_(1)

        return label_mask_i, label_mask_j, label_mask_ij


    def pair_cosine_similarity(self, x, x_adv, eps=1e-8):
        n = x.norm(p=2, dim=1, keepdim=True)
        n_adv = x_adv.norm(p=2, dim=1, keepdim=True)
        return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_adv @ x_adv.t()) / (n_adv * n_adv.t()).clamp(min=eps), (
                    x @ x_adv.t()) / (n * n_adv.t()).clamp(min=eps)

    def nt_xent(self, x, x_adv, label_mask_i, label_mask_j, label_mask_ij, cuda=True):
        '''
        t_mask_i, t_mask_j, t_mask_ij = self.mask_temprature_samples(self.batch_size, c_i, c_j)
        t_mask_i = t_mask_i.detach()
        t_mask_j = t_mask_j.detach()
        t_mask_ij = t_mask_ij.detach()
        '''

        x, x_adv, x_c = self.pair_cosine_similarity(x, x_adv)
        x = torch.exp(x / self.temperature)
        x_adv = torch.exp(x_adv / self.temperature)
        x_c = torch.exp(x_c / self.temperature)
        mask_count_i = label_mask_i.sum(1)
        mask_count_j = label_mask_j.sum(1)
        mask_count_ij = label_mask_ij.sum(1)
        mask_count_ji = label_mask_ij.sum(0)

        mask_count_1 = mask_count_i + mask_count_ij - 1
        mask_count_2 = mask_count_j + mask_count_ji - 1

        #mask_count = torch.cat((mask_count_i, mask_count_j), dim=0)
        #mask_count_2 = torch.cat((mask_count_ij, mask_count_ji), dim=0)
        #mask_count = mask_count + mask_count_2

        mask_reverse = (~(label_mask_ij.bool())).long()
        if cuda:
            dis = (x * (label_mask_i - torch.eye(x.size(0)).long().cuda()) + x_c * label_mask_ij) / (
                        x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / self.temperature))) + mask_reverse
            dis_adv = (x_adv * (label_mask_j - torch.eye(x.size(0)).long().cuda()) + x_c.T * label_mask_ij.T) / (
                        x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / self.temperature))) + mask_reverse.T
        else:
            dis = (x * (label_mask_i - torch.eye(x.size(0)).long()) + x_c * label_mask_ij) / (
                        x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / self.temperature))) + mask_reverse
            dis_adv = (x_adv * (label_mask_j - torch.eye(x.size(0)).long()) + x_c.T * label_mask_ij.T) / (
                        x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / self.temperature))) + mask_reverse.T
        loss = (torch.log(dis).sum(1)) / mask_count_1 + (torch.log(dis_adv).sum(1)) / mask_count_2
        #loss = (torch.log(dis).sum(1)) / mask_count
        # loss = dis.sum(1) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + dis_adv.sum(1) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t)))
        return -loss.mean()
        # return -torch.log(loss).mean()

    def forward(self, z_i, z_j, c_i, c_j):
        self.batch_size = len(z_i)
        #temperature_mask = self.mask_temprature_samples(self.batch_size, c_i, c_j)
        label_mask_i, label_mask_j, label_mask_ij = self.getPseudoLabel(c_i, c_j, threshold=0.7)
        sup_cont_loss = self.nt_xent(z_i, z_j, label_mask_i, label_mask_j, label_mask_ij, cuda=True)
        '''
        if len(z_i) > self.batch_size/2:
            label_mask = self.mask_correlated_samples(preds_i, preds_j)
            sup_cont_loss = self.nt_xent(z_i, z_j, label_mask, cuda=True)
        else:
            return 0
        '''


        '''
        preds_i, preds_j, z_i, z_j = self.getPseudoLabel(z_i, z_j, c_i, c_j, threshold=0)
        label_mask = self.mask_correlated_samples(preds_i, preds_j)
        sup_cont_loss = self.nt_xent(z_i, z_j, c_i, c_j, label_mask, cuda=True)
        '''

        return sup_cont_loss




class MaskInstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(MaskInstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def mask_temprature_samples(self, batch_size, c_i, c_j):
        c = torch.cat((c_i, c_j), dim=0)
        t_mask = torch.matmul(c, c.T)
        t_mask[t_mask < self.temperature] = self.temperature
        for i in range(batch_size):
            t_mask[i, batch_size + i] = self.temperature
            t_mask[batch_size + i, i] = self.temperature
        #logger.debug(t_mask)

        return t_mask

    def forward(self, z_i, z_j, c_i, c_j):
        self.batch_size = len(z_i)
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        temperature_mask = self.mask_temprature_samples(self.batch_size, c_i, c_j)
        temperature_mask = temperature_mask.detach()
        #sim = torch.matmul(z, z.T) / self.temperature
        sim = torch.matmul(z, z.T) / temperature_mask
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        #print(positive_samples)
        #print(negative_samples)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss




class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        self.batch_size = len(z_i)
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        #print(positive_samples)
        #print(negative_samples)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss





class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss
