import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

# from models.diffusion.arinv_model import ARInvModel
from models.diffusion.helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    mask_loss_mean,
    Losses,
)

class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, cond_step, lane_num, observation_dim, action_dim, eta=1, n_timesteps=100,
        sample_steps=1, loss_type='l1', clip_denoised=False, predict_epsilon=True, hidden_dim=256,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1, ar_inv=False, train_only_inv=False, use_unet=False):
        super().__init__()

        self.horizon = horizon
        self.cond_step = cond_step
        self.lane_num = lane_num
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.ar_inv = ar_inv
        self.train_only_inv = train_only_inv
        self.use_unet= use_unet
        if self.ar_inv:
            self.inv_model = None # ARInvModel(hidden_dim=hidden_dim, lane_num=lane_num, observation_dim=observation_dim, action_dim=4)
        else:
            self.inv_model = nn.Sequential(
                nn.Linear(2 * (self.lane_num * self.observation_dim), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4),
            )
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.eta = eta
        self.n_timesteps = int(n_timesteps)
        self.sample_steps = int(sample_steps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_alphas_cumprod_prev', torch.sqrt(alphas_cumprod_prev))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['state_l2'](loss_weights)

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
        '''
        self.action_weight = 1
        dim_weights = torch.ones((self.lane_num, self.observation_dim), dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,lt->hlt', discounts, dim_weights)[None]

        if self.use_unet:
            loss_weights = loss_weights.reshape(*loss_weights.shape[:2], self.lane_num * self.observation_dim)

        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, epsilon_t, t, prev_t):        
        # prev_t = t - self.sample_steps + 1

        sqrt_alphas_cumprod_prev_t = extract(self.sqrt_alphas_cumprod_prev, prev_t + 1, epsilon_t.shape)
        alphas_cumprod_t = extract(self.alphas_cumprod, t, epsilon_t.shape)
        alphas_cumprod_prev_t = extract(self.alphas_cumprod_prev, prev_t + 1, epsilon_t.shape)

        posterior_sigma = self.eta * torch.sqrt(
            ((1 - alphas_cumprod_prev_t) / (1 - alphas_cumprod_t)) * (1 - alphas_cumprod_t / alphas_cumprod_prev_t))
        posterior_dir = torch.sqrt(1 - alphas_cumprod_prev_t - posterior_sigma ** 2)

        posterior_mean = (
            sqrt_alphas_cumprod_prev_t * x_start +
            posterior_dir * epsilon_t
        )

        return posterior_mean, posterior_dir, posterior_sigma
    
        # posterior_mean = (
        #     extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
        #     extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        # )
        # posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        # posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        # posterior_log_variance_clipped = (0.5 * posterior_log_variance_clipped).exp()
        # return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, neighbor_x, cond, t, prev_t, returns=None, returns_mask=None):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, neighbor_x, t, returns, returns_mask, drop_return=False)
            epsilon_uncond = self.model(x, neighbor_x, t, returns, returns_mask, drop_return=True)
            epsilon_cond = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)

            if not self.use_unet:
                returns_mask = returns_mask[..., None]

            epsilon = returns_mask * epsilon_cond + (1 - returns_mask) * epsilon_uncond
        else:
            epsilon = self.model(x, neighbor_x, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_dir, posterior_sigma = self.q_posterior(
                x_start=x_recon, epsilon_t=epsilon, t=t, prev_t=prev_t)
        return model_mean, posterior_dir, posterior_sigma, x_recon

    @torch.no_grad()
    def p_sample(self, x, neighbor_x, cond, t, prev_t, returns=None, returns_mask=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_sigma, x_recon = self.p_mean_variance(x=x, neighbor_x=neighbor_x, cond=cond, t=t, prev_t=prev_t, returns=returns, returns_mask=returns_mask)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * model_sigma * noise, x_recon

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, neighbor_info=None, obs_mask=None, drop_dcm=False, drop_prcd=False, verbose=True, return_diffusion=False):
        device = self.betas.device
        # neighbor_state_idx, neighbor_state_idx_mask = neighbor_info
        lane_neighbors_idx, lane_neighbors_idx_mask = neighbor_info
        b_, n_ ,t_, l_, f_, c_, h_, n_l_ = *shape, self.cond_step, self.horizon, lane_neighbors_idx.shape[1]

        x = 0.5 * torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.cond_step, obs_mask)

        x = x.permute(0, 1, 3, 2, 4).reshape(b_, n_ * l_, t_, f_)
        cond = cond.permute(0, 1, 3, 2, 4).reshape(b_, n_ * l_, c_, f_)

        # neighbor_x
        neighbor_x = cond[:, lane_neighbors_idx] * lane_neighbors_idx_mask[None, ..., None, None]
        neighbor_x = torch.cat((neighbor_x, torch.zeros((*neighbor_x.shape[:3], h_ - c_, f_), device=neighbor_x.device)), dim=-2)

        x = x.reshape(b_ * n_, l_, t_, f_).permute(0, 2, 1, 3)
        neighbor_x = neighbor_x.reshape(b_ * n_, l_, n_l_, h_, f_).permute(0, 1, 3, 2, 4)
        cond = cond.reshape(b_ * n_, l_, c_, f_).permute(0, 2, 1, 3)
        returns = returns.reshape(-1, *returns.shape[2:], 1)
        obs_mask = obs_mask.reshape(-1, *obs_mask.shape[2:])

        if self.use_unet:
            x = x.reshape(b_ * n_, t_, l_ * f_)
            cond = cond.reshape(b_ * n_, c_, l_ * f_)

        if not drop_prcd:
            returns_mask = obs_mask[..., None]
        else:
            returns_mask = torch.ones_like(obs_mask[..., None], device=device)

        if return_diffusion: diffusion = [x]

        t_seq = list(range(0, self.n_timesteps, self.sample_steps))
        prev_t_seq = [-1] + t_seq[:-1]
        for i, j in zip(reversed(t_seq), reversed(prev_t_seq)):
            timesteps = torch.full((x.shape[0],), i, device=device, dtype=torch.long)
            prev_timesteps = torch.full((x.shape[0],), j, device=device, dtype=torch.long)
            x, x_recon = self.p_sample(x, neighbor_x, cond, timesteps, prev_timesteps, returns, returns_mask)
            x = apply_conditioning(x, cond, self.cond_step, obs_mask, self.use_unet)

            if not drop_dcm and not self.use_unet and i < (self.n_timesteps - 1):
                x_recon = apply_conditioning(x_recon, cond, self.cond_step, obs_mask, self.use_unet)
                neighbor_x = x_recon.reshape(b_, n_, t_, l_, f_).permute(0, 1, 3, 2, 4).reshape(b_, n_ * l_, t_, f_)
                neighbor_x = neighbor_x[:, lane_neighbors_idx] * lane_neighbors_idx_mask[None, ..., None, None]
                neighbor_x = neighbor_x.reshape(b_ * n_, l_, n_l_, h_, f_).permute(0, 1, 3, 2, 4)

            if return_diffusion: diffusion.append(x)

        x = x.reshape(shape)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, *args, **kwargs):
        device = self.betas.device
        shape = (*cond.shape[:2], self.horizon, self.lane_num, self.observation_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, t, returns=None,  neighbor_info=None, mask=None, drop_dcm=False, drop_prcd=False):
        lane_neighbors_idx, lane_neighbors_idx_mask = neighbor_info
        obs_mask, cond_mask = mask
        b_, n_ ,t_, l_, f_, = x_start.shape
        n_l_, h_ = lane_neighbors_idx.shape[1], self.horizon

        x_start = x_start.permute(0, 1, 3, 2, 4).reshape(b_, n_ * l_, t_, f_)

        x_noise = torch.randn_like(x_start)
        t = t.flatten()

        # neighbor_x
        neighbor_x = x_start[:, lane_neighbors_idx]
        neighbor_x = neighbor_x.reshape(b_ * n_, l_, n_l_, h_, f_).permute(0, 1, 3, 2, 4)
        lane_neighbors_idx_mask = lane_neighbors_idx_mask.reshape(1, n_, l_, 1, n_l_, 1) \
                                    .repeat(b_, 1, 1, 1, 1, 1).reshape(b_ * n_, l_, 1, n_l_, 1)
        neighbor_x = neighbor_x * lane_neighbors_idx_mask

        # x
        x_start = x_start.reshape(b_ * n_, l_, t_, f_).permute(0, 2, 1, 3)
        x_noise = x_noise.reshape(b_ * n_, l_, t_, f_).permute(0, 2, 1, 3)

        if self.use_unet:
            x_start = x_start.reshape(b_ * n_, t_, l_ * f_)
            x_noise = x_noise.reshape(b_ * n_, t_, l_ * f_)

        returns = returns.reshape(-1, *returns.shape[2:], 1)
        cond_mask = cond_mask.reshape(-1, *cond_mask.shape[2:])
        obs_mask = obs_mask.reshape(-1, *obs_mask.shape[2:])
        x_mask = cond_mask * obs_mask
        if not drop_prcd:
            returns_mask = x_mask[..., None]
        else:
            returns_mask = torch.ones_like(x_mask[..., None], device=x_mask.device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=x_noise)
        x_noisy = apply_conditioning(x_noisy, x_start, self.cond_step, x_mask, self.use_unet)

        x_recon = self.model(x_noisy, neighbor_x, t, returns, returns_mask)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, x_start, self.cond_step, x_mask, self.use_unet)

        assert x_noise.shape == x_recon.shape

        loss_mask = 1 - cond_mask
        loss_mask[..., self.cond_step:] = obs_mask[..., self.cond_step:]

        if self.use_unet:
            loss_mask = loss_mask[..., None]
        else:
            loss_mask = loss_mask[..., None, None]

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, x_noise, loss_mask)
        else:
            loss, info = self.loss_fn(x_recon, x_start, loss_mask)

        return loss, info

    def loss(self, batch_traj_mask, returns=None, neighbor_info=None, drop_dcm=False, drop_prcd=False):
        x, a, obs_mask, cond_mask = batch_traj_mask

        if self.train_only_inv:
            loss = self.train_inv_model(x, a, obs_mask)
            info = {'inv_loss':loss}
        else:
            batch_size = x.shape[:2]
            t = torch.randint(0, self.n_timesteps, batch_size, device=x.device).long()
            diffuse_loss, info = self.p_losses(x, t, returns, neighbor_info, [obs_mask, cond_mask], drop_dcm, drop_prcd)

            inv_loss = self.train_inv_model(x, a, obs_mask)
            loss = (1 / 2) * (diffuse_loss + inv_loss)

        return loss, info
    
    def train_inv_model(self, x, a, obs_mask):
        # Calculating inv loss
        x_t, x_t_1 = x[..., :-1, :, :], x[..., 1:, :, :]
        obs_mask_t, obs_mask_t_1 = obs_mask[..., :-1], obs_mask[..., 1:]
        a_t = a[..., :-1]

        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        x_comb_t = x_comb_t.reshape(-1, 2 * (self.lane_num * self.observation_dim))
        a_t = a_t.reshape(-1, self.action_dim)
        action_mask_t = (obs_mask_t * obs_mask_t_1).flatten()

        if self.ar_inv:
            inv_loss = self.inv_model.calc_loss(x_comb_t, a_t)
        else:
            pred_a_t = self.inv_model(x_comb_t)
            # one-hot encode the action
            target_a_t =  torch.zeros_like(pred_a_t).scatter(1, a_t.long(), 1)

            inv_loss = F.cross_entropy(pred_a_t, target_a_t, reduction='none')
            inv_loss = mask_loss_mean(inv_loss, action_mask_t)

        return inv_loss

    
    def reset_inv_model(self):
        # init inv model
        for p in self.inv_model.parameters():
            if p.dim() > 1:
                stdv = 1. / math.sqrt(p.size(1))
                p.data.uniform_(-stdv, stdv)
            else:
                p.data.uniform_(-stdv, stdv)


    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)