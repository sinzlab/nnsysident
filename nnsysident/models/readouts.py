import torch
import numpy as np
from torch import nn
from nnfabrik.utility.nn_helpers import get_module_output
from torch.nn import Parameter
from torch.nn import functional as F

from mlutils.layers.readouts import PointPooled2d, FullGaussian2d, SpatialXFeatureLinear, ConfigurationError
from mlutils.constraints import positive


class MultiReadout:

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        l1_reg = self.gamma_readout * self[data_key].feature_l1(average=False)
        return l1_reg


class MultipleFullGaussian2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, init_mu_range, init_sigma, bias, gamma_readout,
                 gauss_type, grid_mean_predictor, grid_mean_predictor_type, source_grids,
                 share_features, share_grid, share_transform, shared_match_ids, init_noise, init_transform_scale):
        # super init to get the _module attribute
        super().__init__()
        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            source_grid = None
            shared_grid = None
            shared_transform = None
            if grid_mean_predictor is not None:
                if grid_mean_predictor_type == 'cortex':
                    source_grid = source_grids[k]
                else:
                    raise KeyError('grid mean predictor {} does not exist'.format(grid_mean_predictor_type))
                if share_transform:
                    shared_transform = None if i == 0 else self[k0].mu_transform

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
                source_grid=source_grid,
                shared_transform=shared_transform,
                init_noise=init_noise,
                init_transform_scale=init_transform_scale,
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

    def __init__(self, in_shape, outdims, bias, grid_mean_predictor=None, shared_features=None, source_grid=None,  **kwargs):

        super().__init__()
        self.in_shape = in_shape
        c, w, h = in_shape
        self.outdims = outdims

        if grid_mean_predictor is None:
            self._mu = Parameter(data=torch.zeros(outdims, 2), requires_grad=True)
        else:
            self.init_grid_predictor(source_grid=source_grid, **grid_mean_predictor)

        self.log_var = Parameter(
            data=torch.zeros(outdims, 2), requires_grad=True)
        self.grid = torch.nn.Parameter(
            data=self.make_mask_grid(), requires_grad=False)

        self.initialize_features(**(shared_features or {}))

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)

        self.initialize()

    @property
    def shared_features(self):
        return self._features

    @property
    def features(self):
        if self._shared_features:
            return self.scales * self._features[..., self.feature_sharing_index]
        else:
            return self._features


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

    @property
    def mu(self):
        if self._predicted_grid:
            return self.mu_transform(self.source_grid.squeeze())#.view(*self.grid_shape)
        else:
            return self._mu

    def init_grid_predictor(self, source_grid, hidden_features=20, hidden_layers=0, final_tanh=False):
        self._original_grid = False
        layers = [
            nn.Linear(source_grid.shape[1], hidden_features if hidden_layers > 0 else 2)
        ]

        for i in range(hidden_layers):
            layers.extend([
                nn.ELU(),
                nn.Linear(hidden_features, hidden_features if i < hidden_layers - 1 else 2)
            ])

        if final_tanh:
            layers.append(
                nn.Tanh()
            )
        self.mu_transform = nn.Sequential(*layers)

        source_grid = source_grid - source_grid.mean(axis=0, keepdims=True)
        source_grid = source_grid / np.abs(source_grid).max()
        self.register_buffer('source_grid', torch.from_numpy(source_grid.astype(np.float32)))
        self._predicted_grid = True

    def initialize_features(self, match_ids=None, shared_features=None):
        """
        The internal attribute `_original_features` in this function denotes whether this instance of the FullGuassian2d
        learns the original features (True) or if it uses a copy of the features from another instance of FullGaussian2d
        via the `shared_features` (False). If it uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        c, w, h = self.in_shape
        self._original_features = True
        if match_ids is not None:
            assert self.outdims == len(match_ids)

            n_match_ids = len(np.unique(match_ids))
            if shared_features is not None:
                assert shared_features.shape == (1, c, 1, n_match_ids), \
                    f'shared features need to have shape (1, {c}, 1, {n_match_ids})'
                self._features = shared_features
                self._original_features = False
            else:
                self._features = Parameter(
                    torch.Tensor(1, c, 1, n_match_ids))  # feature weights for each channel of the core
            self.scales = Parameter(torch.Tensor(1, 1, 1, self.outdims))  # feature weights for each channel of the core
            _, sharing_idx = np.unique(match_ids, return_inverse=True)
            self.register_buffer('feature_sharing_index', torch.from_numpy(sharing_idx))
            self._shared_features = True
        else:
            self._features = Parameter(
                torch.Tensor(1, c, 1, self.outdims))  # feature weights for each channel of the core
            self._shared_features = False

    def initialize(self):
        """
        initialize function initializes the mean, sigma for the Gaussian readout and features weights
        """
        self._features.data.fill_(1 / self.in_shape[0])
        if self._shared_features:
            self.scales.data.fill_(1.)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def feature_l1(self, average=True):
        """
        Returns the l1 regularization term either the mean or the sum of all weights
        Args:
            average(bool): if True, use mean of weights for regularization

        """
        if self._original_features:
            if average:
                return self._features.abs().mean()
            else:
                return self._features.abs().sum()
        else:
            return 0

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
    def __init__(self, core, in_shape_dict, n_neurons_dict, bias, gamma_readout, grid_mean_predictor, grid_mean_predictor_type, share_features, source_grids, shared_match_ids):
        # super init to get the _module attribute
        super(MultipleDeterministicgaussian2d, self).__init__()

        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            source_grid = None
            if grid_mean_predictor is not None:
                if grid_mean_predictor_type == 'cortex':
                    source_grid = source_grids[k]
                else:
                    raise KeyError('grid mean predictor {} does not exist'.format(grid_mean_predictor_type))

            if share_features:
                shared_features = {
                    'match_ids': shared_match_ids[k],
                    'shared_features': None if i == 0 else self[k0].shared_features
                }
            else:
                shared_features = None

            self.add_module(k, DeterministicGaussian2d(
                in_shape=in_shape,
                outdims=n_neurons,
                bias=bias,
                grid_mean_predictor=grid_mean_predictor,
                shared_features=shared_features,
                source_grid=source_grid
            )
                            )
        self.gamma_readout = gamma_readout

    def regularizer(self, data_key):
        return (self[data_key].feature_l1(average=False) + self[data_key].variance_l1(average=False)) * self.gamma_readout

#####################################################################################


class FullSXF(nn.Module):

    def __init__(self, in_shape, outdims, bias, normalize=True, init_noise=1e-3, shared_features=None, **kwargs):

        super().__init__()

        c, w, h = in_shape
        self.in_shape = in_shape
        self.outdims = outdims

        self.init_noise = init_noise
        self.normalize = normalize

        self._original_features = True
        self.initialize_features(**(shared_features or {}))
        self.spatial = Parameter(torch.Tensor(self.outdims, w, h))

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.initialize()

    @property
    def shared_features(self):
        return self._features

    @property
    def features(self):
        if self._shared_features:
            return self.scales * self._features[self.feature_sharing_index, ...]
        else:
            return self._features

    @property
    def weight(self):
        n = self.outdims
        c, w, h = self.in_shape
        return self.normalized_spatial.view(n, 1, w, h) * self.features.view(n, c, 1, 1)

    @property
    def normalized_spatial(self):
        positive(self.spatial)
        if self.normalize:
            norm = self.spatial.pow(2).sum(dim=1, keepdim=True)
            norm = norm.sum(dim=2, keepdim=True).sqrt().expand_as(self.spatial) + 1e-6
            weight = self.spatial / norm
        else:
            weight = self.spatial
        return weight

    def l1(self, average=False):
        n = self.outdims
        c, w, h = self.in_shape

        if self._original_features:
            ret = (self.normalized_spatial.view(self.outdims, -1).abs().sum(dim=1, keepdim=True)
                   * self.features.view(self.outdims, -1).abs().sum(dim=1)).sum()
            if average:
                ret = ret / (n * c * w * h)
        else:
            ret = self.normalized_spatial.view(self.outdims, -1).abs().sum()
            if average:
                ret = ret / (n * w * h)
        return ret

    def initialize(self):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """
        self.spatial.data.normal_(0, self.init_noise)
        self._features.data.normal_(0, self.init_noise)
        #self._features.data.fill_(1 / self.in_shape[0])
        if self._shared_features:
            self.scales.data.fill_(1.)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def initialize_features(self, match_ids=None, shared_features=None):
        """
        The internal attribute `_original_features` in this function denotes whether this instance of the FullGuassian2d
        learns the original features (True) or if it uses a copy of the features from another instance of FullGaussian2d
        via the `shared_features` (False). If it uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        c, w, h = self.in_shape
        if match_ids is not None:
            assert self.outdims == len(match_ids)

            n_match_ids = len(np.unique(match_ids))
            if shared_features is not None:
                assert shared_features.shape == (n_match_ids, c), \
                    f'shared features need to have shape ({n_match_ids}, {c})'
                self._features = shared_features
                self._original_features = False
            else:
                self._features = Parameter(
                    torch.Tensor(n_match_ids, c))  # feature weights for each channel of the core
            self.scales = Parameter(torch.Tensor(self.outdims, 1))  # feature weights for each channel of the core
            _, sharing_idx = np.unique(match_ids, return_inverse=True)
            self.register_buffer('feature_sharing_index', torch.from_numpy(sharing_idx))
            self._shared_features = True
        else:
            self._features = Parameter(
                torch.Tensor(self.outdims, c))  # feature weights for each channel of the core
            self._shared_features = False

    def forward(self, x, shift=None):
        N, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError("the specified feature map dimension is not the readout's expected input dimension")

        y = torch.einsum('ncwh,owh->nco', x, self.normalized_spatial)
        y = torch.einsum('nco,oc->no', y, self.features)
        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        c, w, h = self.in_shape
        r = self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.outdims) + ")"
        if self.bias is not None:
            r += " with bias"
        if self._shared_features:
            r += ", with {} features".format('original' if self._original_features else 'shared')
        if self.normalize:
            r += ", normalized"
        else:
            r += ", unnormalized"
        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r


class MultipleFullSXF(MultiReadout, torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, init_noise, bias, normalize, gamma_readout, share_features, shared_match_ids):
        # super init to get the _module attribute
        super().__init__()
        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            if share_features:
                shared_features = {
                    'match_ids': shared_match_ids[k],
                    'shared_features': None if i == 0 else self[k0].shared_features
                }
            else:
                shared_features = None

            self.add_module(k, FullSXF(
                in_shape=in_shape,
                outdims=n_neurons,
                bias=bias,
                normalize=normalize,
                init_noise=init_noise,
                shared_features=shared_features,
            )
                            )
        self.gamma_readout = gamma_readout

    def regularizer(self, data_key):
        return self[data_key].l1(average=False) * self.gamma_readout
