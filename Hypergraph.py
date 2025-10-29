import torch
import torch.nn as nn
import torch.nn.functional as nnF


class TanhReLU(nn.Module):
    def __init__(self):
        super(TanhReLU, self).__init__()

    def forward(self, x):
        return torch.tanh(nnF.relu(x))


class HypergraphConvolution(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.P = nn.Parameter(torch.randn([F, F], requires_grad=True))
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Parameter):
            nn.init.xavier_normal_(m)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, H, x, W = None):
        if W is not None:
            x = nn.functional.elu(H @ W @ H.t() @ x @ self.P)
        else:
            x = nn.functional.elu(H @ H.t() @ x @ self.P)
        return x



def keep_top_k_in_rows(matrix, k):
    if k >= matrix.shape[1]:
        return matrix


    _, top_indices = torch.topk(matrix, k, dim=1)


    mask = torch.zeros_like(matrix, dtype=torch.bool)


    mask.scatter_(1, top_indices, torch.ones_like(top_indices, dtype=torch.bool))


    result = matrix.clone()


    result[~mask] = float('-inf')

    return result

class GenerateLocalHypergraph(nn.Module):
    def __init__(self, N, T, F, num_heads_CausalMHA = 1, Kn=64, num_Local_HGConv=1,
                 num_local_hyperedge=128, dropout=0.3, epsilon=1e-6, device='cuda'):
        super().__init__()
        self.CausalMHA = nn.MultiheadAttention(embed_dim=F, num_heads=num_heads_CausalMHA, dropout = dropout,
                                               bias = True, batch_first = True)

        self.T = T
        self.N = N
        self.F = F
        self.Kn = Kn

        indices = torch.arange(T * N, device=device)
        row_block = (indices // N).unsqueeze(1)
        col_block = (indices // N).unsqueeze(0)
        self.attention_mask = row_block < col_block

        self.H_embedding1 = nn.Linear(T * N, T * N)
        self.H_embedding2 = nn.Linear(T * N, num_local_hyperedge)

        self.epsilon = epsilon
        self.LocalHGConv = nn.ModuleList([HypergraphConvolution(F) for _ in range(num_Local_HGConv)])

        self.layernorm = nn.LayerNorm(F)

        self.apply(self.init_weights)
        self.device = device

        self.norm = nn.ModuleList([nn.LayerNorm(F) for _ in range(num_Local_HGConv)])
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_Local_HGConv)])

        self.retu = TanhReLU()

        self.dropout0 = nn.Dropout(dropout)

    def init_weights(self, m):
        if isinstance(m, nn.Parameter):
            nn.init.xavier_normal_(m)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = x.transpose(0, 1).flatten(end_dim=1)
        _, attn_weights_TN = self.CausalMHA(x, x, x, attn_mask=self.attention_mask)

        attn_weights_TN_topK = keep_top_k_in_rows(attn_weights_TN, self.Kn)

        attn_weights_TN_topK = nn.functional.softmax(attn_weights_TN_topK, dim=1)



        H = self.dropout0(self.retu(self.H_embedding1(attn_weights_TN_topK)))
        H = self.retu(self.H_embedding2(H))


        for i, m in enumerate(self.LocalHGConv):
            residual = x
            out = self.norm[i](x)
            out = m(H=H, x=out)
            out = self.dropout[i](out)
            x = residual + out

        x = x.reshape(self.T, self.N, self.F).transpose(0, 1)

        return x, attn_weights_TN_topK, H


def jensen_shannon_divergence(matrix):

    epsilon = 1e-6
    matrix = matrix / (torch.sum(matrix, dim=0, keepdim=True) + epsilon)

    # Get shapes
    N, M = matrix.shape

    P = matrix.unsqueeze(2).expand(N, M, M)
    Q = matrix.unsqueeze(1).expand(N, M, M)

    mixture = 0.5 * (P + Q)


    kl_p_m = torch.sum(P * torch.log2((P + epsilon) / (mixture + epsilon)), dim=0)
    kl_q_m = torch.sum(Q * torch.log2((Q + epsilon) / (mixture + epsilon)), dim=0)

    # Calculate JSD
    jsd_matrix = 0.5 * kl_p_m + 0.5 * kl_q_m

    return jsd_matrix


class GenerateGlobalHypergraph(nn.Module):
    def __init__(self, T, F, num_global_hyperedge, num_Global_HGConv=1, dropout=0.1, epsilon=1e-6):
        super().__init__()
        self.linear1 = nn.Linear(T * F, T * F)
        self.linear2 = nn.Linear(T * F, num_global_hyperedge)

        self.dropout0 = nn.Dropout(dropout)
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_Global_HGConv)])
        self.epsilon = epsilon
        self.HGConv = nn.ModuleList([HypergraphConvolution(T*F) for _ in range(num_Global_HGConv)])
        self.apply(self.init_weights)

        self.norm = nn.ModuleList([nn.LayerNorm(T*F) for _ in range(num_Global_HGConv)])

        self.retu = TanhReLU()

    def init_weights(self, m):
        if isinstance(m, nn.Parameter):
            nn.init.xavier_normal_(m)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, g):
        x = x.flatten(start_dim=1)
        g = g.flatten(start_dim=1)

        g = self.dropout0(self.retu(self.linear1(g)))  #
        g = self.retu(self.linear2(g))

        prob_g = nn.functional.softmax(g, dim=0)

        jsd = jensen_shannon_divergence(prob_g)
        mean_jsd = jsd.mean(dim=0)
        normalize_mean_jsd = (mean_jsd - mean_jsd.mean()) / (mean_jsd.std() + self.epsilon)
        W_ = nn.functional.softmax(normalize_mean_jsd, dim=0)
        W = W_.diag()

        for i, m in enumerate(self.HGConv):
            residual = x
            out = self.norm[i](x)
            out = m(H=prob_g, x=out, W = W)
            out = self.dropout[i](out)
            x = residual + out

        return x, prob_g, W_