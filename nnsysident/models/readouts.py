import torch

from torch import nn
from nnfabrik.utility.nn_helpers import get_module_output
from torch.nn import Parameter

from mlutils.layers.readouts import PointPooled2d, FullGaussian2d, SpatialXFeatureLinear


class MultiReadout:

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultipleFullGaussian2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, init_mu_range, init_sigma, bias, gamma_readout,
                 gauss_type, grid_mean_predictor, grid_mean_predictor_type, source_grids,
                 share_features, share_grid, shared_match_ids):
        # super init to get the _module attribute
        super().__init__()
        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            source_grid = None
            shared_grid = None
            if grid_mean_predictor is not None:
                if grid_mean_predictor_type == 'cortex':
                    source_grid = source_grids[k]
                else:
                    raise KeyError('grid mean predictor {} does not exist'.format(grid_mean_predictor_type))
            elif share_grid:
                shared_grid = {
                    'match_ids': shared_match_ids[k],
                    'shared_grid': None if i == 0 else self[k0].shared_grid
                }

            if share_features:
                shared_features = {
                    'match_ids': shared_match_ids[k],
                    'shared_features': None if i == 0 else self[k0].shared_features
                }
            else:
                shared_features = None

            self.add_module(k, FullGaussian2d(
                in_shape=in_shape,
                outdims=n_neurons,
                init_mu_range=init_mu_range,
                init_sigma=init_sigma,
                bias=bias,
                gauss_type=gauss_type,
                grid_mean_predictor=grid_mean_predictor,
                shared_features=shared_features,
                shared_grid=shared_grid,
                source_grid=source_grid
            )
                            )
        self.gamma_readout = gamma_readout


class MultiplePointPooled2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, pool_steps, pool_kern, bias, init_range, gamma_readout):
        # super init to get the _module attribute
        super(MultiplePointPooled2d, self).__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            self.add_module(k, PointPooled2d(
                in_shape,
                n_neurons,
                pool_steps=pool_steps,
                pool_kern=pool_kern,
                bias=bias,
                init_range=init_range)
                            )
        self.gamma_readout = gamma_readout


class MultipleSpatialXFeatureLinear(MultiReadout, torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, init_noise, bias, normalize, gamma_readout):
        # super init to get the _module attribute
        super().__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            self.add_module(k, SpatialXFeatureLinear(
                in_shape=in_shape,
                outdims=n_neurons,
                init_noise=init_noise,
                bias=bias,
                normalize=normalize
            )
                            )
        self.gamma_readout = gamma_readout

    def regularizer(self, data_key):
        return self[data_key].l1(average=False) * self.gamma_readout


########################################################################################################################

class DeterministicGaussian2d(nn.Module):
    """
    'Gaussian2d' class instantiates an object that can be used to learn a
    Normal distribution in the core feature space for each neuron, this is
    applied to the pixel grid to give a simple weighting.
    The variance should be decreased over training to achieve localization.
    The readout receives the shape of the core as 'in_shape', the number of units/neurons being predicted as 'outdims', 'bias' specifying whether
    or not bias term is to be used.
    The grid parameter contains the normalized locations (x, y coordinates in the core feature space) and is clipped to [-1.1] as it a
    requirement of the torch.grid_sample function. The feature parameter learns the best linear mapping between the feature
    map from a given location, and the unit's response with or without an additional elu non-linearity.
    Args:
        in_shape (list): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
    """

    def __init__(self, in_shape, outdims, bias, **kwargs):

        super().__init__()
        self.in_shape = in_shape
        c, w, h = in_shape
        self.outdims = outdims

        self.mu = Parameter(
            data=torch.zeros(outdims, 2), requires_grad=True)
        self.log_var = Parameter(
            data=torch.zeros(outdims, 2), requires_grad=True)
        self.grid = torch.nn.Parameter(
            data=self.make_mask_grid(), requires_grad=False)

        self.features = Parameter(torch.Tensor(1, c, 1, outdims))  # saliency  weights for each channel from core

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)

        self.initialize()

    def make_mask_grid(self):
        xx, yy = torch.meshgrid(
            [torch.linspace(-1, 1, self.in_shape[1]),
             torch.linspace(-1, 1, self.in_shape[2])])
        grid = torch.stack([xx, yy], 2)[None, ...]
        return grid.repeat([self.outdims, 1, 1, 1])

    def mask(self):
        variances = torch.exp(self.log_var).view(self.outdims, 1, 1, -1)
        mean = self.mu.view(self.outdims, 1, 1, -1)
        pdf = self.grid - mean
        pdf = torch.sum((pdf**2) / variances, dim=-1)
        pdf = torch.exp(-.5 * pdf)
        # normalize to sum=1
        pdf = pdf / torch.sum(pdf, dim=(1, 2), keepdim=True)
        return pdf

    def initialize(self):
        """
        initialize function initializes the mean, sigma for the Gaussian readout and features weights
        """
        self.features.data.fill_(1 / self.in_shape[0])

        if self.bias is not None:
            self.bias.data.fill_(0)

    def feature_l1(self, average=True):
        """
        feature_l1 function returns the l1 regularization term either the mean or just the sum of weights
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        if average:
            return self.features.abs().mean()
        else:
            return self.features.abs().sum()

    def variance_l1(self, average=True):
        """
        feature_l1 function returns the l1 regularization term either the
        mean or just the sum of variances
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        if average:
            return self.log_var.abs().mean()
        else:
            return self.log_var.abs().sum()

    def forward(self, x, out_idx=None):
        N, c, w, h = x.size()
        feat = self.features.view(1, c, self.outdims)

        if out_idx is None:
            bias = self.bias
            outdims = self.outdims
        else:
            feat = feat[:, :, out_idx]
            if self.bias is not None:
                bias = self.bias[out_idx]
            outdims = len(out_idx)

        mask = self.mask()
        mask = mask.permute(1, 2, 0)[None, None, ...]

        y = torch.einsum("bchw, jkhwn-> bcn", x, mask)
        y = (y * feat).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        return y

    def __repr__(self):
        c, w, h = self.in_shape
        r = self.__class__.__name__ + \
            ' (' + '{} x {} x {}'.format(c, w, h) + ' -> ' + str(self.outdims) + ')'
        if self.bias is not None:
            r += ' with bias'
        for ch in self.children():
            r += '  -> ' + ch.__repr__() + '\n'
        return r


class MultipleDeterministicgaussian2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, bias, gamma_readout):
        # super init to get the _module attribute
        super(MultipleDeterministicgaussian2d, self).__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            self.add_module(k, DeterministicGaussian2d(
                in_shape,
                n_neurons,
                bias=bias)
                            )
        self.gamma_readout = gamma_readout
