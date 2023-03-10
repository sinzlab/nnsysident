import torch

from mei.legacy.utils import varargin


class ChangeNormAndClip:
    """Change the norm of the input.
    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, norm, x_min, x_max):
        self.norm = norm
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x.view(len(x), -1), dim=-1)
        renorm = x * (self.norm / x_norm).view(len(x), *[1] * (x.dim() - 1))
        return torch.clamp(renorm, self.x_min, self.x_max)


class ChangeStdClampedMean:
    """Change the norm of the input.
    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, std, x_min, x_max, clamped_mean):
        self.clamped_mean = clamped_mean
        self.std = std
        self.x_min = x_min
        self.x_max = x_max
        self.clamped_mean = clamped_mean

    @varargin
    def __call__(self, x, iteration=None):
        x = x.clamp(self.x_min, self.x_max)
        x_std = torch.std(x.view(len(x), -1), dim=-1)

        # set x to have the desired std
        x = x * (self.std / (x_std + 1e-9)).view(len(x), *[1] * (x.dim() - 1))
        # compute mean of x
        x_mean = torch.mean(x.view(len(x), -1), dim=-1)
        # set mean to the clamped value
        x = x + (self.clamped_mean - x_mean).view(len(x), *[1] * (x.dim() - 1))
        return x
