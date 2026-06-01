# coding: UTF-8
import torch
import torch.nn as nn


class SVFLinear(nn.Module):

    def __init__(self, base_linear, r=16, alpha=32, dropout=0.1):
        super(SVFLinear, self).__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")

        self.base_linear = base_linear
        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.in_features = int(base_linear.in_features)
        self.out_features = int(base_linear.out_features)

        max_rank = min(self.out_features, self.in_features)
        self.r = min(int(r), max_rank)
        if self.r <= 0:
            raise ValueError(f"Invalid SVF rank: {r}")

        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(float(dropout))

        with torch.no_grad():
            W0 = self.base_linear.weight.detach().float().cpu()
            U, _, Vh = torch.linalg.svd(W0, full_matrices=False)
            U_r = U[:, :self.r].contiguous()
            V_r = Vh[:self.r, :].T.contiguous()

        self.register_buffer("U", U_r)
        self.register_buffer("V", V_r)
        self.delta_sigma = nn.Parameter(torch.zeros(self.r, self.r, dtype=torch.float32))

    def forward(self, x):
        base_output = self.base_linear(x)
        U = self.U.to(device=x.device, dtype=x.dtype)
        V = self.V.to(device=x.device, dtype=x.dtype)
        delta_sigma = self.delta_sigma.to(dtype=x.dtype)
        svf_output = self.dropout(x) @ V @ delta_sigma.T @ U.T
        return base_output + self.scaling * svf_output

    @torch.no_grad()
    def set_delta_sigma(self, delta_sigma):
        """Overwrite local DeltaSigma with a server-broadcast update."""
        if not torch.is_tensor(delta_sigma):
            delta_sigma = torch.tensor(delta_sigma)
        if tuple(delta_sigma.shape) != tuple(self.delta_sigma.shape):
            raise ValueError(
                f"delta_sigma shape mismatch: expected {tuple(self.delta_sigma.shape)}, "
                f"got {tuple(delta_sigma.shape)}"
            )
        self.delta_sigma.copy_(delta_sigma.to(device=self.delta_sigma.device, dtype=self.delta_sigma.dtype))

    @torch.no_grad()
    def reset_delta_sigma(self):
        self.delta_sigma.zero_()

    def get_delta_sigma(self, detach=True, cpu=True):
        x = self.delta_sigma
        if detach:
            x = x.detach()
        if cpu:
            x = x.cpu()
        return x

    def get_svf_state(self, detach=True, cpu=True):
        U = self.U
        V = self.V
        delta_sigma = self.delta_sigma
        if detach:
            U = U.detach()
            V = V.detach()
            delta_sigma = delta_sigma.detach()
        if cpu:
            U = U.cpu()
            V = V.cpu()
            delta_sigma = delta_sigma.cpu()
        return {
            "U": U,
            "V": V,
            "delta_sigma": delta_sigma,
            "r": self.r,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "scaling": self.scaling,
        }

    def get_delta_weight(self, scaled=True):
        delta_w = self.U @ self.delta_sigma @ self.V.T
        if scaled:
            delta_w = self.scaling * delta_w
        return delta_w
