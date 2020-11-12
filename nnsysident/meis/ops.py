import torch
from mei.legacy.utils import varargin


class ChangeNormAndClip:
    """ Change the norm of the input.
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
