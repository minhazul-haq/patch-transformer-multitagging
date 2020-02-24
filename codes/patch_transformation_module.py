# author: Mohammad Minhazul Haq
# created on: March 3, 2020

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTransformationModule(nn.Module):
    def __init__(self):
        super(PatchTransformationModule, self).__init__()

        self.H = 3  # number of attention heads
        self.D = 2048  # number of features from resnet-50

        self.attention = AttentionModule()

        self.W = torch.nn.Parameter(torch.randn(self.H * self.D, self.D)) # 3*2048x2048
        self.W.requires_grad = True

        self.relu = nn.ReLU(inplace=True)

    def forward(self, V):
        f1 = self.attention(V)  # MxD
        f2 = self.attention(V)
        f3 = self.attention(V)

        f_concat = torch.cat([f1, f2, f3], dim=1)  # Mx3D
        f_concat_weighted = torch.mm(f_concat, self.W)  # MxD

        V_prime = self.relu(V + f_concat_weighted)

        return V_prime


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()

        self.D_prime = 512
        self.D = 2048  # number of features from resnet-50

        self.U_t = torch.nn.Parameter(torch.randn(self.D_prime, self.D))
        self.U_t.required_grad = True

        self.W_a = torch.nn.Parameter(torch.randn(1, self.D_prime))
        self.W_a.requires_grad = True

        self.tanh = nn.Tanh()


    def forward(self, V):
        tanh_Ut_V = self.tanh(torch.mm(self.U_t, torch.transpose(V, 0, 1)))  # D_prime x M
        a = F.softmax(torch.mm(self.W_a, tanh_Ut_V), dim=1)  # 1xM
        a_transposed = torch.transpose(a, 0, 1)  # Mx1

        A = a_transposed.repeat(1, self.D)  # MxD
        f = torch.mul(V, A)  # MxD

        return f
