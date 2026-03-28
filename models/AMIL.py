import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from .DynamicFilter import DynamicFilterQuant


class AMIL_layer(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.1, activation=None):
        super(AMIL_layer, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        if activation == 'sigmoid':
            self.attention_c = nn.Sequential(
                nn.Linear(D, 1),
                nn.Sigmoid()
            )
        elif activation == 'tanh':
            self.attention_c = nn.Sequential(
                nn.Linear(D, 1),
                nn.Tanh()
            )
        elif activation is None:
            self.attention_c = nn.Linear(D, 1)
        else:
            raise NotImplementedError

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        a = self.attention_a(x)
        b = self.attention_b(x)
        A_without_softmax = a.mul(b)
        A_without_softmax = self.attention_c(
            A_without_softmax).squeeze(dim=2)  
        A = F.softmax(A_without_softmax, dim=1).unsqueeze(dim=1)
        h = torch.bmm(A, x).squeeze(dim=1)
        return h, A.squeeze(dim=1), A_without_softmax

