import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stablemodels.lyapunov import LyapunovFunction

class StableDynamicsModel(nn.Module):
    def __init__(self,
                 input_shape,
                 control_size=None,
                 device = None,
                 alpha=0.9,
                 layer_sizes=[64, 64],
                 lr=3e-4,
                 lyapunov_lr=3e-4,
                 lyapunov_eps=1e-3):
        super(StableDynamicsModel, self).__init__()
        input_shape = np.prod(input_shape)
        if control_size is None:
            control_size = 0
        self._layer_sizes = layer_sizes
        self._lr = lr
        self.device = device
        self._alpha = alpha
        self._lyapunov_function = LyapunovFunction((input_shape + control_size,),
                                                   layer_sizes=layer_sizes + [1],
                                                   lr=lyapunov_lr,
                                                   eps=lyapunov_eps)
        self._l1 = nn.Linear(input_shape + control_size, 64)
        self._l2 = nn.Linear(64, 64)
        self._l3 = nn.Linear(64, input_shape)
        self._obs_size = input_shape
        self._act_size = control_size

    def forward(self, x, u=None):
        xu = x.squeeze()
        if self._act_size != 0:
            xu = torch.cat([x, u], dim=-1).squeeze() # [bs, x_dim+u_dim]
            xu_mask = torch.cat([torch.ones(size=x.size()), torch.zeros(size=u.size())], dim=-1).squeeze().to(self.device)
        if not xu.requires_grad:
            xu.requires_grad = True
        # import pdb; pdb.set_trace()
        f = self._l1(xu.squeeze()).relu()
        f = self._l2(f).relu()
        f = self._l3(f)
        if len(xu.shape) == 1:
            xu = xu.unsqueeze(0)
        lyapunov = self._lyapunov_function(xu * xu_mask).squeeze()  #zeros?ones?
        xu.retain_grad()
        lyapunov.backward(gradient=torch.ones_like(lyapunov), retain_graph=True)
        grad_v = xu.grad.clone() # [bs, x_dim+u_dim]
        gv = grad_v.view(-1, 1, xu.shape[-1])[:,:,:self._obs_size] # [bs, 1, x_dim]

        fv = f.view(-1, self._obs_size, 1)
        dot = (gv @ fv).squeeze()
        orth = (dot + self._alpha * lyapunov).squeeze().relu() / gv.squeeze().pow(2).sum(-1)

        if len(orth.shape) == 0:
            orth.unsqueeze_(0)
        # return f - (torch.diag_embed(orth) @ grad_v)[:, :self._obs_size]
        return f - (orth.reshape(-1,1) * gv.squeeze())[:, :self._obs_size]

    def predict(self, x, u=None):
        x = x.permute(1, 0, 2)
        if u is not None:
            u = u.permute(1, 0, 2)
        prediction = torch.zeros_like(x)
        xt = x[0]
        for t in range(x.shape[0]):
            ut = None if u is None else u[t]
            prediction[t] = self.forward(xt, ut)
            xt = prediction[t]
        return prediction.permute(1, 0, 2)
