import torch
import math
from torch import nn


class SelfAttention2D_Block_N(nn.Module):
    def __init__(self, N, T, F, T_dim, D, dropout=0.3, device='cuda'):
        super(SelfAttention2D_Block_N, self).__init__()
        self.Wq = nn.Parameter(torch.randn(T_dim, F, D, device=device).unsqueeze(0).expand(N, -1, -1, -1).reshape(-1, F, D), requires_grad=True)
        self.Wk = nn.Parameter(torch.randn(T_dim, F, D, device=device).unsqueeze(0).expand(N, -1, -1, -1).reshape(-1, F, D), requires_grad=True)
        self.Wv = nn.Parameter(torch.randn(T_dim, F, D, device=device).unsqueeze(0).expand(N, -1, -1, -1).reshape(-1, F, D), requires_grad=True)
        self.bq = nn.Parameter(torch.zeros(T_dim, 1, D, device=device).unsqueeze(0).expand(N, -1, -1, -1).reshape(-1, 1, D), requires_grad=True)
        self.bk = nn.Parameter(torch.zeros(T_dim, 1, D, device=device).unsqueeze(0).expand(N, -1, -1, -1).reshape(-1, 1, D), requires_grad=True)
        self.bv = nn.Parameter(torch.zeros(T_dim, 1, D, device=device).unsqueeze(0).expand(N, -1, -1, -1).reshape(-1, 1, D) , requires_grad=True)

        self._norm_fact = 1.0 / math.sqrt(D)

        self.linear1 = nn.Linear(T_dim * T * T, T_dim)
        self.linear2 = nn.Linear(T_dim, 1)
        self.linear3 = nn.Linear(D * T_dim, D)
        self.linear4 = nn.Linear(D, F)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


        self.T = T
        self.N = N
        self.F = F
        self.T_dim = T_dim
        self.D = D
        self.device = device

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Parameter):
            nn.init.xavier_normal_(m)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):  # x: [N, T, F]
        mat_a = x.unsqueeze(1).expand(-1, self.T_dim, -1, -1).reshape(-1, self.T, self.F)

        q = torch.bmm(mat_a, self.Wq) + self.bq
        k = torch.bmm(mat_a, self.Wk) + self.bk
        v = torch.bmm(mat_a, self.Wv) + self.bv

        # 计算注意力分数
        q_ = q.reshape(self.N, self.T_dim, self.T, self.D).unsqueeze(1).expand(-1, self.N, -1, -1, -1).reshape(-1,
                                                                                                               self.T,
                                                                                                               self.D)
        k_ = k.reshape(self.N, self.T_dim, self.T, self.D).unsqueeze(0).expand(self.N, -1, -1, -1, -1).reshape(-1,
                                                                                                               self.T,
                                                                                                               self.D).transpose(
            2, 1)

        alpha = torch.bmm(q_, k_) * self._norm_fact
        alpha = alpha.reshape(self.N, self.N, self.T_dim * self.T * self.T)
        alpha = self.dropout1(torch.nn.functional.relu(self.linear1(alpha)))
        alpha = self.linear2(alpha).squeeze()
        att_score = torch.nn.functional.softmax(alpha, dim=1)

        v_ = v.reshape(self.N, self.T_dim, self.T, self.D)
        b = torch.einsum('ij,jklm->iklm', att_score, v_)

        b = b.permute(0, 2, 3, 1).reshape(self.N, self.T, self.D * self.T_dim)
        y = self.dropout2(torch.nn.functional.relu(self.linear3(b)))
        y = self.linear4(y)

        return y, att_score


class SelfAttention2D_N(nn.Module):
    def __init__(self, N, T, F, T_dim, D, num_SelfAttention2D_Block_N=1, dropout=0.3):
        super(SelfAttention2D_N, self).__init__()
        self.layers = nn.ModuleList(
            [SelfAttention2D_Block_N(N, T, F, T_dim, D, dropout) for _ in range(num_SelfAttention2D_Block_N)])
        self.layernorm = nn.ModuleList([nn.LayerNorm(F) for _ in range(num_SelfAttention2D_Block_N)])
        self.num_SelfAttention2D_Block_N = num_SelfAttention2D_Block_N
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_SelfAttention2D_Block_N)])

    def forward(self, x):
        att_scores = []
        for i in range(self.num_SelfAttention2D_Block_N):
            residual = x
            out = self.layernorm[i](x)
            out, score = self.layers[i](out)
            att_scores.append(score)
            out = self.dropout[i](out)
            x = residual + out
        return x, att_scores