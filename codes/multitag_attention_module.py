# author: Mohammad Minhazul Haq
# created on: March 4, 2020

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultitagAttentionModule(nn.Module):
    def __init__(self):
        super(MultitagAttentionModule, self).__init__()

        self.D_prime = 512
        self.D = 2048  # number of features from resnet-50
        self.classes = 4

        self.U_t = torch.nn.Parameter(torch.randn(self.D_prime, self.D))
        self.U_t.required_grad = True

        self.W_mta = torch.nn.Parameter(torch.randn(1, self.D_prime))
        self.W_mta.requires_grad = True

        self.W = torch.nn.Parameter(torch.randn(self.classes, self.D))
        self.W.requires_grad = True

        self.tanh = nn.Tanh()


    def forward(self, V_prime):
        tanh_Ut_V_prime = self.tanh(torch.mm(self.U_t, torch.transpose(V_prime, 0, 1)))  # D_prime x M
        a = F.softmax(torch.mm(self.W_mta, tanh_Ut_V_prime), dim=1)  # 1xM
        a_transposed = torch.transpose(a, 0, 1)  # Mx1

        t_mat = torch.mul(a_transposed, V_prime)  # MxD
        #print("in mta forward...")
        #print(t_mat.shape)
        t = torch.transpose((torch.sum(t_mat, 0)).resize(1, self.D), 0, 1)  # D x 1

        l = (torch.mm(self.W, t)).resize(1, self.classes)  # 1 x num_of_classes

        return l
