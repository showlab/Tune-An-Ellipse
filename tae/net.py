from typing import Union, Iterator

import torch
import torch.nn as nn

from tae.ellipse import RotatatedEllipse


class EllipseGenerator(nn.Module):
    def __init__(self,
                 initial_ellipse: Union[list, tuple],   # [cx, cy, a, b, theta]
                 h: int, w: int,
                 mu: float = 1, sigma: float = 0.1, eta: float = 10
                 ):
        super(EllipseGenerator, self).__init__()

        self.init_ellipse = torch.tensor(initial_ellipse)

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 5)
        self._init_weights()

        self.ellipse = RotatatedEllipse(h, w, mu, sigma, eta)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.nn.LeakyReLU(0.1)(self.fc1(x))
        x = torch.nn.LeakyReLU(0.1)(self.fc2(x))
        x = torch.tanh(0.5 * self.fc3(x))

        x = x[0]
        cx = self.init_ellipse[0] + x[0]
        cy = self.init_ellipse[1] + x[1]
        a = self.init_ellipse[2] + x[2]
        b = self.init_ellipse[3] + x[3]
        theta = self.init_ellipse[4] + x[4]

        r_ellipse_curve, r_ellipse_region = self.ellipse(cx, cy, a, b, theta)

        return r_ellipse_curve, r_ellipse_region, (cx, cy, a, b, theta)

