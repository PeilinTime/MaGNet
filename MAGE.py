import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Callable, Union, Tuple
from einops import einsum, rearrange, repeat
import math

class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


def exists(val):
    return val is not None


def default(val, default_val):
    return default_val if val is None else val


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)

class SwiGLU(nn.Module):

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class GLU(nn.Module):

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Callable[[Tensor], Tensor],
        mult_bias: bool = False,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.0

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate) * self.mult_bias


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: Optional[int] = None,
        dim_out: Optional[int] = None,
        mult: Optional[int] = 4,
        glu: Optional[bool] = False,
        glu_mult_bias: Optional[bool] = False,
        swish: Optional[bool] = False,
        relu_squared: Optional[bool] = False,
        post_act_ln: Optional[bool] = False,
        dropout: Optional[float] = 0.0,
        no_bias: Optional[bool] = False,
        zero_init_output: Optional[bool] = False,
        custom_act: Optional[nn.Module] = None,
        swiglu: Optional[bool] = False,
        triton_kernels_on: bool = False,
    ):

        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.mult = mult
        self.glu = glu
        self.glu_mult_bias = glu_mult_bias
        self.swish = swish
        self.relu_squared = relu_squared
        self.post_act_ln = post_act_ln
        self.dropout = dropout
        self.no_bias = no_bias
        self.zero_init_output = zero_init_output
        self.custom_act = custom_act
        self.swiglu = swiglu
        self.triton_kernels_on = triton_kernels_on

        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        elif custom_act is not None:
            activation = custom_act
        elif swiglu:
            activation = SwiGLU()
        else:
            activation = nn.GELU()

        if glu:
            project_in = GLU(
                dim, inner_dim, activation, mult_bias=glu_mult_bias
            )

        else:
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias=not no_bias), activation
            )

        if post_act_ln:
            self.ff = nn.Sequential(
                project_in,
                nn.LayerNorm(inner_dim),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out, bias=no_bias),
            )
        else:
            self.ff = nn.Sequential(
                project_in,
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out, bias=not no_bias),
            )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)


class SwitchGate(nn.Module):

    def __init__(
        self,
        dim,
        num_experts: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor, use_aux_loss=False):
        gate_scores = F.softmax(self.w_gate(x), dim=-1)

        capacity = int(self.capacity_factor * x.size(0))

        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)

        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )

        masked_gate_scores = gate_scores * mask

        denominators = (
            masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        )

        gate_scores = (masked_gate_scores / denominators) * capacity

        if use_aux_loss:
            load = gate_scores.sum(0)
            importance = gate_scores.sum(1)

            loss = ((load - importance) ** 2).mean()

            return gate_scores, loss

        return gate_scores, None


class SwitchMoE(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        mult: int = 4,
        use_aux_loss: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList(
            [
                FeedForward(dim, dim, mult, *args, **kwargs)
                for _ in range(num_experts)
            ]
        )

        self.gate = SwitchGate(
            dim,
            num_experts,
            capacity_factor,
        )

    def forward(self, x: Tensor):
        gate_scores, loss = self.gate(
            x, use_aux_loss=self.use_aux_loss
        )

        expert_outputs = [expert(x) for expert in self.experts]

        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0

        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )

        return moe_output, loss


class MambaBlock(nn.Module):

    def __init__(
        self,
        dim: int = None,
        depth: int = 5,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias

        dt_rank = math.ceil(self.dim / 16)
        self.dt_rank = dt_rank

        dim_inner = dim * expand
        self.dim_inner = dim_inner

        self.in_proj = nn.Linear(dim, dim_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=dim_inner,
            out_channels=dim_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=dim_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(
            dim_inner, dt_rank + self.d_state * 2, bias=False
        )

        self.dt_proj = nn.Linear(dt_rank, dim_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=dim_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(dim_inner))
        self.out_proj = nn.Linear(dim_inner, dim, bias=bias)

    def forward(self, x: Tensor):
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        x_and_res = rearrange(x_and_res, "b l x -> b x l")
        (x, res) = x_and_res.split(
            split_size=[self.dim_inner, self.dim_inner], dim=1
        )

        x = self.conv1d(x)[:, :, :l]
        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(rearrange(y, "b dim l -> b l dim"))

        return output

    def ssm(self, x: Tensor):
        (d_in, n) = self.A_log.shape


        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = rearrange(x, "b d l -> b l d")
        x_dbl = self.x_proj(x_dbl)

        (delta, B, C) = x_dbl.split(
            split_size=[self.dt_rank, n, n], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))

        y = self.selective_scan(
            x, delta, A, B, C, D
        )

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, d_in, l) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b d_in l n"))
        deltaB_u = einsum(
            delta, B, u, "b l d_in, b l n, b d_in l -> b d_in l n"
        )

        x = torch.zeros((b, d_in, n), device=next(self.parameters()).device)
        ys = []
        for i in range(l):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            y = einsum(x, C[:, i, :], "b d_in n , b n -> b d_in")
            ys.append(y)
        y = torch.stack(ys, dim=2)  # (b d_in l)

        if D is not None:
            y = y + u * rearrange(D, "d_in -> d_in 1")

        return y


class RMSNorm(nn.Module):

    def __init__(self, dim, groups=1):
        super().__init__()
        self.scale = dim**-0.5
        self.gamma = nn.Parameter(torch.ones(groups, dim, 1))

    def forward(self, x):
        normed = F.normalize(x, dim=-2)
        return normed * self.scale * self.gamma


class MambaMoEGRUAttentionBlock(nn.Module):
    def __init__(
            self,
            T: int,
            dim: int,
            depth: int,
            d_state: int,
            dropout: float = 0.1,
            m_expand: int = 2,
            num_experts: int = 4,
            gru_layer: int = 1,
            gru_bidirectional: bool = False,
            num_heads_mha: int = 1,
    ):
        super().__init__()
        self.depth = depth
        self.T = T
        self.dim = dim
        self.depth = depth
        self.d_state = d_state
        self.m_expand = m_expand
        self.num_experts = num_experts
        self.hidden_dim = dim * m_expand

        self.mamba_forward = nn.ModuleList([
            MambaBlock(
                dim=dim,
                depth=1,
                d_state=d_state,
                expand=m_expand,
                d_conv=4,
                conv_bias=True,
                bias=False,
            )
            for _ in range(depth)
        ])

        self.mamba_backward = nn.ModuleList([
            MambaBlock(
                dim=dim,
                depth=1,
                d_state=d_state,
                expand=m_expand,
                d_conv=4,
                conv_bias=True,
                bias=False,
            )
            for _ in range(depth)
        ])

        self.MoE = nn.ModuleList([
            SwitchMoE(
                dim=dim,
                hidden_dim=self.hidden_dim,
                output_dim=dim,
                num_experts=num_experts,
            )
            for _ in range(depth)
        ])

        self.gru = nn.ModuleList([nn.GRU(input_size=dim, hidden_size=dim,
                                         num_layers=gru_layer, bias=True,
                                         batch_first=True, dropout=dropout,
                                         bidirectional=gru_bidirectional) for _ in range(depth)])

        self.mha = nn.ModuleList([nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads_mha, dropout=dropout,
                                                        bias=True, batch_first=True)
                                  for _ in range(depth)])

        self.norm1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])

        self.dropout1 = nn.ModuleList([nn.Dropout(dropout) for _ in range(depth)])
        self.dropout2 = nn.ModuleList([nn.Dropout(dropout) for _ in range(depth)])
        self.dropout3 = nn.ModuleList([nn.Dropout(dropout) for _ in range(depth)])

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        for i in range(self.depth):
            residual = x
            out = self.norm1[i](x)
            mamba_forward = self.mamba_forward[i](out)
            mamba_backward = self.mamba_backward[i](out.flip([1])).flip([1]).transpose(0, 1).contiguous()
            outputs = []
            for t in range(self.T):
                out_, _ = self.gru[i](mamba_forward[:, t:t+1], mamba_backward[t:t+1])
                outputs.append(out_)
            out = torch.cat(outputs, dim=1)
            out = self.dropout1[i](out)
            x = residual + out

            residual = x
            out = self.norm2[i](x)
            out, _ = self.MoE[i](out)
            out = self.dropout2[i](out)
            x = residual + out

            residual = x
            out = self.norm3[i](x)
            out, _ = self.mha[i](out, out, out)
            out = self.dropout3[i](out)
            x = residual + out

        return x