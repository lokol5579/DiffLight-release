import math

import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from torchinfo import summary


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CommAttention(nn.Module):
    def __init__(self, dim, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        hidden_dim = dim * num_heads

        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(dim, hidden_dim, bias=False)

        self.q_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(dim, hidden_dim),
        )
        self.k_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(dim, hidden_dim),
        )
        self.v_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(dim, hidden_dim),
        )

        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, q_x, kv_x, t_embed):
        """_summary_

        Args:
            q (_type_): (b, t, l, f)
            kv (_type_): (b, n, t, l', f)
            t_embed (_type_): (b, t, 1, 2f)
        """

        b, t, l, f = q_x.shape
        b_, n_, t_, l_, f_ = kv_x.shape

        assert n_ == l

        q_x = q_x.permute(0, 2, 1, 3).reshape(b * l, t, f)
        kv_x = kv_x.reshape(b * n_, t_, l_, f)

        q_t_embed = t_embed.repeat(1, l, 1, 1).reshape(b * l, t, -1)
        kv_t_embed = t_embed[:, :t_].repeat(1, n_, 1, 1).reshape(b * n_, t_, 1, -1)
        
        q = (self.to_q(q_x) + self.q_time_mlp(q_t_embed)).reshape(b * l, t, self.num_heads, self.dim)
        k = (self.to_k(kv_x) + self.k_time_mlp(kv_t_embed)).reshape(b * n_, -1, self.num_heads, self.dim)
        v = (self.to_v(kv_x) + self.v_time_mlp(kv_t_embed)).reshape(b * n_, -1, self.num_heads, self.dim)

        q = q.permute(0, 2, 1, 3).reshape(b * l * self.num_heads, t, self.dim)
        k = k.permute(0, 2, 1, 3).reshape(b * n_ * self.num_heads, -1, self.dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_ * self.num_heads, -1, self.dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim)
        scores = scores.softmax(dim=-1)
        
        out = torch.matmul(scores, v).reshape(b, l, self.num_heads, t, self.dim)
        out = out.permute(0, 3, 1, 2, 4).reshape(b, t, l, self.num_heads * self.dim)

        return self.to_out(out)
        

class SpatioAttention(nn.Module):
    def __init__(self, dim, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        hidden_dim = dim * num_heads

        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(dim, hidden_dim, bias=False)

        self.q_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(dim, hidden_dim),
        )
        self.k_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(dim, hidden_dim),
        )
        self.v_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(dim, hidden_dim),
        )

        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, qkv_x, t_embed, reward_embed):
        """_summary_

        Args:
            q (_type_): (b, t, l, f)
            kv (_type_): (b, t, l', f)
            t_embed (_type_): (b, t, 1, 2f)
            reward_embed (_type_): (b, t, 1, f)
        """

        b, t, l, f = qkv_x.shape

        qkv_x = qkv_x.reshape(b * t, l, f)

        if reward_embed is None:
            t_r_embed = t_embed
        else:
            t_r_embed = t_embed + reward_embed
        t_r_embed = t_r_embed.reshape(b * t, 1, -1)

        q = (self.to_q(qkv_x) + self.q_time_mlp(t_r_embed)).reshape(b * t, l, self.num_heads, self.dim)
        k = (self.to_k(qkv_x) + self.k_time_mlp(t_r_embed)).reshape(b * t, l, self.num_heads, self.dim)
        v = (self.to_v(qkv_x) + self.v_time_mlp(t_r_embed)).reshape(b * t, l, self.num_heads, self.dim)

        q = q.permute(0, 2, 1, 3).reshape(b * t * self.num_heads, l, self.dim)
        k = k.permute(0, 2, 1, 3).reshape(b * t * self.num_heads, l, self.dim)
        v = v.permute(0, 2, 1, 3).reshape(b * t * self.num_heads, l, self.dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim)
        scores = scores.softmax(dim=-1)
        
        out = torch.matmul(scores, v).reshape(b, t, self.num_heads, l, self.dim)
        out = out.permute(0, 1, 3, 2, 4).reshape(b, t, l, self.num_heads * self.dim)

        return self.to_out(out)
    

class TemperalAttention(nn.Module):
    def __init__(self, dim, num_heads=3, cond_step=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        hidden_dim = dim * num_heads
        self.cond_step = cond_step

        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(dim, hidden_dim, bias=False)

        self.q_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(dim, hidden_dim),
        )
        self.k_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(dim, hidden_dim),
        )
        self.v_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(dim, hidden_dim),
        )

        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, qkv_x, t_embed, reward_embed):
        """_summary_

        Args:
            q (_type_): (b, t, l, f)
            kv (_type_): (b, t, l', f)
        """

        b, t, l, f = qkv_x.shape

        qkv_x = qkv_x.permute(0, 2, 1, 3).reshape(b * l, t, f)

        if reward_embed is None:
            t_r_embed = t_embed
        else:
            t_r_embed = t_embed + reward_embed
        t_r_embed = t_r_embed.permute(0, 2, 1, 3).repeat(1, l, 1, 1).reshape(b * l, t, -1)

        q = (self.to_q(qkv_x) + self.q_time_mlp(t_r_embed)).reshape(b * l, t, self.num_heads, self.dim)
        k = (self.to_k(qkv_x) + self.k_time_mlp(t_r_embed)).reshape(b * l, t, self.num_heads, self.dim)
        v = (self.to_v(qkv_x) + self.v_time_mlp(t_r_embed)).reshape(b * l, t, self.num_heads, self.dim)

        q = q.permute(0, 2, 1, 3).reshape(b * l * self.num_heads, t, self.dim)
        k = k.permute(0, 2, 1, 3).reshape(b * l * self.num_heads, t, self.dim)
        v = v.permute(0, 2, 1, 3).reshape(b * l * self.num_heads, t, self.dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim)
        scores = scores.softmax(dim=-1)
        
        out = torch.matmul(scores, v).reshape(b, l, self.num_heads, t, self.dim)
        out = out.permute(0, 3, 1, 2, 4).reshape(b, t, l, self.num_heads * self.dim)

        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=3, n_groups=8, cond_step=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=n_groups, num_channels=dim)
        self.norm2 = nn.GroupNorm(num_groups=n_groups, num_channels=dim)
        self.norm3 = nn.GroupNorm(num_groups=n_groups, num_channels=dim)
        self.norm4 = nn.GroupNorm(num_groups=n_groups, num_channels=dim)
        self.norm5 = nn.GroupNorm(num_groups=n_groups, num_channels=dim)

        self.comm_attn = CommAttention(dim, num_heads)
        self.temp_attn = TemperalAttention(dim, num_heads, cond_step)
        self.spat_attn = SpatioAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, neighbor_x, t_embed, reward_embed):
        if neighbor_x is not None:
            assert x.shape[2] == neighbor_x.shape[1]
            comm_val = self.comm_attn(self.norm1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1), self.norm2(neighbor_x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1), t_embed=t_embed)
            x = x + comm_val

        spat_val = self.spat_attn(self.norm3(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1), t_embed=t_embed, reward_embed=reward_embed)
        temp_val = self.temp_attn(self.norm4(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1), t_embed=t_embed, reward_embed=reward_embed)
        x = x + spat_val + temp_val

        x = self.norm5(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = x + self.mlp(x)
        
        return x



class STFormer(nn.Module):
    def __init__(self,
                 horizon,
                 cond_step,
                 transition_dim,
                 hidden_dim=128,
                 block_depth=3,
                 reward_condition=False,
                 condition_dropout=0.3,
                 drop_neighbor=False,
                 kernel_size=3,
                 ):
        super().__init__()

        self.horizon = horizon
        self.cond_step = cond_step
        self.step_time_dim = hidden_dim
        self.diffuse_time_dim = hidden_dim

        self.reward_dim = hidden_dim
        self.reward_condition = reward_condition
        self.condition_dropout = condition_dropout
        self.drop_neighbor = drop_neighbor

        act_fn = nn.Mish()

        self.diffuse_time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            act_fn,
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.step_time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            act_fn,
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.x_mlp = nn.Sequential(
            nn.Linear(transition_dim, hidden_dim * 4),
            act_fn,
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.neighbor_x_mlp = nn.Sequential(
            nn.Linear(transition_dim, hidden_dim * 4),
            act_fn,
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        if reward_condition:
            self.reward_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, hidden_dim * 4),
                act_fn,
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            self.mask_dist = Bernoulli(probs=1 - condition_dropout)

        self.former_blocks = nn.ModuleList()
        for i in range(block_depth):
            self.former_blocks.append(TransformerBlock(dim=hidden_dim, num_heads=3, cond_step=cond_step))

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            act_fn,
            nn.Linear(hidden_dim * 4, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, transition_dim),
        )

    def forward(self, x, neighbor_x, diffuse_time, reward=None, reward_mask=None, drop_return=False):
        # add diffuse time, step time and reward
        diffuse_t_embed = self.diffuse_time_mlp(diffuse_time)
        diffuse_t_embed = diffuse_t_embed.reshape(diffuse_t_embed.shape[0], 1, 1, diffuse_t_embed.shape[1])
        diffuse_t_embed = diffuse_t_embed.repeat(1, x.shape[1], 1, 1)

        step_time = torch.arange(self.horizon).to(x.device)
        step_t_embed = self.step_time_mlp(step_time)
        step_t_embed = step_t_embed.reshape(1, step_t_embed.shape[0], 1, step_t_embed.shape[1])
        step_t_embed = step_t_embed.repeat(x.shape[0], 1, 1, 1)

        t_embed = diffuse_t_embed + step_t_embed

        if not drop_return and self.reward_condition:
            reward_embed = self.reward_mlp(reward)
            reward_embed = reward_embed * reward_mask
            if self.training:
                reward_embed = reward_embed * self.mask_dist.sample((reward_embed.shape[0], 1, 1)).to(reward_embed.device)
            reward_embed = reward_embed.reshape(reward_embed.shape[0], reward_embed.shape[1], 1, reward_embed.shape[2])
            # reward_embed[:, :self.cond_step] = 0
        else:
            reward_embed = None

        # input mlp
        x = self.x_mlp(x)
        neighbor_x = None if self.drop_neighbor else self.neighbor_x_mlp(neighbor_x)

        # former blocks
        for i, block in enumerate(self.former_blocks):
            x = block(x, neighbor_x, t_embed=t_embed, reward_embed=reward_embed)

        # output mlp
        x = self.out_mlp(x)

        return x
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    st_former = STFormer(
        horizon=8,
        cond_step=5,
        transition_dim=2,
        hidden_dim=64,
        block_depth=1,
        reward_condition=True,
        condition_dropout=0.1,
    ).to(device=device)
    x = torch.ones(2, 8, 12, 2).to(device=device)
    neighbor_x = torch.ones(2, 12, 8, 12, 2).to(device=device)
    diffuse_time = torch.ones(2,).to(device=device)
    reward = torch.ones(2, 8, 1).to(device=device)
    reward_mask = torch.ones(2, 8, 1).to(device=device)
    cond_step = 5
    summary(st_former, input_data=[x, neighbor_x, diffuse_time, reward, reward_mask])
