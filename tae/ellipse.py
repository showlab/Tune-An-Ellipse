"""
author: sierkinhane@gmail.com
2023-10-10 18:54
"""
from typing import Union, List

import numpy as np
import torch
from matplotlib import pyplot as plt

torch_pi = np.pi


class RotatatedEllipse(torch.nn.Module):
    def __init__(self, h, w, mu=1, sigma=0.05, eta=10):
        super(RotatatedEllipse, self).__init__()
        with torch.no_grad():
            self.grid_x, self.grid_y = torch.meshgrid([torch.linspace(0, 1, w),
                                                       torch.linspace(0, 1, h)])
            self.grid_x = self.grid_x.view(w, h)
            self.grid_y = self.grid_y.view(w, h)
        self.mu = mu
        self.sigma = sigma  # could be learnable/adaptive
        self.init_sigma = sigma
        self.eta = eta
        self.target_size = w

    def _gaussian(self, x):
        return torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)

    def support_batch_ellipse(self, cx, cy, a, b, theta):
        if len(cx.shape) == 0:
            return self.grid_x, self.grid_y, cx, cy, a, b, theta
        batch_size = cx.shape[0]
        grid_x = self.grid_x.unsqueeze(0).repeat(batch_size, 1, 1)
        grid_y = self.grid_y.unsqueeze(0).repeat(batch_size, 1, 1)
        cx = cx[:, None, None]
        cy = cy[:, None, None]
        a = a[:, None, None]
        b = b[:, None, None]
        theta = theta[:, None, None]
        return grid_x, grid_y, cx, cy, a, b, theta

    def sync_device(self, cx, cy, a, b, theta):
        if self.grid_x.device != cx.device:
            self.grid_x = self.grid_x.to(cx.device)
            self.grid_y = self.grid_y.to(cx.device)

    def forward(self, cx, cy, a, b, theta):
        self.sync_device(cx, cy, a, b, theta)
        grid_x, grid_y, cx, cy, a, b, theta = self.support_batch_ellipse(cx, cy, a, b, theta)

        # to adaptively control the width of the curve
        self.sigma = (-(a + b) / 2 + 1) * self.init_sigma
        left_part = ((grid_x - cx) * torch.cos(torch_pi * theta) + (grid_y - cy) * torch.sin(
            torch_pi * theta)) ** 2 / (a ** 2)
        right_part = ((grid_x - cx) * torch.sin(torch_pi * theta) - (grid_y - cy) * torch.cos(
            torch_pi * theta)) ** 2 / (b ** 2)

        r_ellipse_curve = self._gaussian(left_part + right_part)

        r_ellipse_region = 0.5 + (1 / torch_pi) * torch.atan(self.eta * (left_part + right_part) - self.eta)

        return r_ellipse_curve, 1 - r_ellipse_region


class MultiRotElpse:
    def __init__(self, target_sizes: Union[List[int], int], mu=1, sigma=0.05, eta=10):
        if isinstance(target_sizes, int):
            target_sizes = [target_sizes]
        self.ellipses = {
            target_size: RotatatedEllipse(target_size, target_size, mu, sigma, eta)
            for target_size in target_sizes
        }

    def __getitem__(self, target_size):
        return self.ellipses[target_size]

