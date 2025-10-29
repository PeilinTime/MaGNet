from torch import nn
from F2DAttn import SelfAttention2D_F
from S2DAttn import SelfAttention2D_N
from Hypergraph import GenerateLocalHypergraph, GenerateGlobalHypergraph
from MAGE import MambaMoEGRUAttentionBlock

class MaGNet(nn.Module):
    def __init__(self, N, T, F, dim, num_MAGE, num_experts,
                 num_heads_mha, num_F2DAttn, num_channels,
                 num_heads_CausalMHA, num_TCH, TopK, M1,
                 num_S2DAttn, num_GPH, M2, device, dropout=0.1):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(F, F),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(F, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            )

        self.mambamoegrumha = MambaMoEGRUAttentionBlock(
            T=T,
            dim=dim,
            depth=num_MAGE,
            d_state=16,
            dropout = dropout,
            m_expand = 4,
            num_experts = num_experts,
            gru_layer = 1,
            gru_bidirectional = False,
            num_heads_mha = num_heads_mha)

        self.SelfAttention2D_F = SelfAttention2D_F(N=N, T=T, F=dim, N_dim=num_channels, D=T,
                                                   num_SelfAttention2D_Block_F=num_F2DAttn,
                                                   dropout=dropout)

        self.SelfAttention2D_N = SelfAttention2D_N(N=N, T=T, F=dim, T_dim=num_channels, D=dim,
                                                   num_SelfAttention2D_Block_N=num_S2DAttn, dropout=dropout)

        self.LocalHypergraph = GenerateLocalHypergraph(N=N, T=T, F=dim, num_heads_CausalMHA=num_heads_CausalMHA,
                                                       Kn=TopK, num_Local_HGConv=num_TCH,
                                                       num_local_hyperedge=M1, dropout=dropout,device = device)

        self.GlobalHypergraph = GenerateGlobalHypergraph(T=T, F=dim, num_global_hyperedge=M2,
                                                         num_Global_HGConv=num_GPH, dropout=dropout)
        # FFN output
        self.output = nn.Sequential(
            nn.LayerNorm(T*dim),
            nn.Linear(T*dim, T*dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(T*dim, 2),
        )


    def forward(self, x):
        x = self.embedding(x)

        x = self.mambamoegrumha(x)

        x, att_scores_F = self.SelfAttention2D_F(x)

        x, attn_weights_TN_topK, H_local = self.LocalHypergraph(x)

        g, att_scores_N = self.SelfAttention2D_N(x)

        x, H_gocal, W = self.GlobalHypergraph(x, g)

        x= self.output(x)

        return x, att_scores_F, attn_weights_TN_topK, H_local, H_gocal, W