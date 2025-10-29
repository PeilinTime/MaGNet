import torch
import math
from torch import nn

class SelfAttention2D_Block_F(nn.Module):
    def __init__(self, N, T, F, N_dim, D, dropout=0.1, device='cuda'):
        super(SelfAttention2D_Block_F, self).__init__()

        self.Wq = nn.Parameter(torch.randn(N_dim, T, D, device=device).unsqueeze(0).expand(F, -1, -1, -1).reshape(-1, T, D), requires_grad=True)
        self.Wk = nn.Parameter(torch.randn(N_dim, T, D, device=device).unsqueeze(0).expand(F, -1, -1, -1).reshape(-1, T, D), requires_grad=True)
        self.Wv = nn.Parameter(torch.randn(N_dim, T, D, device=device).unsqueeze(0).expand(F, -1, -1, -1).reshape(-1, T, D), requires_grad=True)
        self.bq = nn.Parameter(torch.zeros(N_dim, 1, D, device=device).unsqueeze(0).expand(F, -1, -1, -1).reshape(-1, 1, D), requires_grad=True)
        self.bk = nn.Parameter(torch.zeros(N_dim, 1, D, device=device).unsqueeze(0).expand(F, -1, -1, -1).reshape(-1, 1, D), requires_grad=True)
        self.bv = nn.Parameter(torch.zeros(N_dim, 1, D, device=device).unsqueeze(0).expand(F, -1, -1, -1).reshape(-1, 1, D), requires_grad=True)

        self._norm_fact = 1.0 / math.sqrt(D)

        self.linear1 = nn.Linear(N_dim * N * N, N_dim)
        self.linear2 = nn.Linear(N_dim, 1)

        # output projections
        self.linear3 = nn.Linear(D * N_dim, D)
        self.linear4 = nn.Linear(D, T)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.N = N
        self.T = T
        self.F = F
        self.N_dim = N_dim
        self.D = D

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Parameter):
            nn.init.xavier_normal_(m)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = x.permute(2, 0, 1)

        mat_a = x.unsqueeze(1).expand(-1, self.N_dim, -1, -1).reshape(-1, self.N, self.T)

        q = torch.bmm(mat_a, self.Wq) + self.bq
        k = torch.bmm(mat_a, self.Wk) + self.bk
        v = torch.bmm(mat_a, self.Wv) + self.bv

        q_ = q.reshape(self.F, self.N_dim, self.N, self.D)
        q_ = q_.unsqueeze(1).expand(-1, self.F, -1, -1, -1).reshape(-1, self.N, self.D)
        k_ = k.reshape(self.F, self.N_dim, self.N, self.D)
        k_ = k_.unsqueeze(0).expand(self.F, -1, -1, -1, -1).reshape(-1, self.N, self.D)

        alpha = torch.bmm(q_, k_.transpose(1, 2)) * self._norm_fact
        alpha = alpha.reshape(self.F, self.F, self.N_dim * self.N * self.N)
        alpha = self.dropout1(torch.relu(self.linear1(alpha)))
        alpha = self.linear2(alpha).squeeze(-1)
        att_score = torch.nn.functional.softmax(alpha, dim=1)

        v_ = v.reshape(self.F, self.N_dim, self.N, self.D)
        b = torch.einsum('ij,jklm->iklm', att_score, v_)

        b = b.permute(0, 2, 3, 1).reshape(self.F, self.N, self.D * self.N_dim)
        y = self.dropout2(torch.relu(self.linear3(b)))
        y = self.linear4(y)

        return y.permute(1, 2, 0), att_score


class SelfAttention2D_F(nn.Module):
    def __init__(self, N, T, F, N_dim, D, num_SelfAttention2D_Block_F=1, dropout=0.3):
        super(SelfAttention2D_F, self).__init__()
        self.layers = nn.ModuleList(
            [SelfAttention2D_Block_F(N, T, F, N_dim, D, dropout) for _ in range(num_SelfAttention2D_Block_F)])
        self.layernorm = nn.ModuleList([nn.LayerNorm(F) for _ in range(num_SelfAttention2D_Block_F)])
        self.num_SelfAttention2D_Block_F = num_SelfAttention2D_Block_F
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_SelfAttention2D_Block_F)])

    def forward(self, x):
        att_scores = []
        for i in range(self.num_SelfAttention2D_Block_F):
            residual = x
            out = self.layernorm[i](x)
            out, score = self.layers[i](out)
            att_scores.append(score)
            out = self.dropout[i](out)
            x = residual + out
        return x, att_scores


