import torch


# maximize similarity
class SimMaxLoss(torch.nn.Module):

    def __init__(self, margin=0):
        super(SimMaxLoss, self).__init__()
        self.margin = margin

    def forward(self, x, weights=1):
        x = x.clamp(0.0001, 0.9999)
        return -(torch.log(x + self.margin) * weights).mean()


# minimize similarity
class SimMinLoss(torch.nn.Module):

    def __init__(self, margin=0):
        super(SimMinLoss, self).__init__()
        self.margin = margin

    def forward(self, x, weights=1):
        x = x.clamp(0.0001, 0.9999)
        return -(torch.log(1 - x + self.margin) * weights).mean()
