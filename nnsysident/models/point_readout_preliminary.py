import logging
import warnings

import numpy as np
import torch
from torch.nn import Parameter
from torch.nn import functional as F

logger = logging.getLogger(__name__)

from neuralpredictors.layers.readouts import PointPooled2d


class GeneralizedPointPooled2d(PointPooled2d):
    def __init__(self, in_shape, outdims, pool_steps, bias, pool_kern, init_range, inferred_params_n=1, **kwargs):
        """
        This is the generalized version of the PointPooled2d which is built to lean any distribution (not only
        Poisson). Specify the number of parameters of the distribution with the argument `inferred_params_n`.
        """

        self.inferred_params_n = inferred_params_n
        super().__init__(in_shape, outdims, pool_steps, bias, pool_kern, init_range, **kwargs)

        c, w, h = in_shape
        self.features = Parameter(torch.Tensor(self.inferred_params_n, 1, c * (self._pool_steps + 1), 1, outdims))

        if bias:
            bias = Parameter(torch.Tensor(inferred_params_n, outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.initialize(self.mean_activity)

    @property
    def pool_steps(self):
        return self._pool_steps

    @pool_steps.setter
    def pool_steps(self, value):
        assert value >= 0 and int(value) - value == 0, "new pool steps must be a non-negative integer"
        if value != self._pool_steps:
            logger.info("Resizing readout features")
            c, w, h = self.in_shape
            self._pool_steps = int(value)
            self.features = Parameter(
                torch.Tensor(self.inferred_params_n, 1, c * (self._pool_steps + 1), 1, self.outdims)
            )
            self.features.data.fill_(1 / self.in_shape[0])

    def forward(self, x, shift=None, out_idx=None, **kwargs):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
            shift: shifts the location of the grid (from eye-tracking data)
            out_idx: index of neurons to be predicted

        Returns:
            y: neuronal activity
        """
        self.grid.data = torch.clamp(self.grid.data, -1, 1)
        N, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if [c_in, w_in, h_in] != [c, w, h]:
            raise ValueError("the specified feature map dimension is not the readout's expected input dimension")

        m = self.pool_steps + 1  # the input feature is considered the first pooling stage
        feat = self.features.view(self.inferred_params_n, 1, m * c, self.outdims)
        if out_idx is None:
            grid = self.grid
            bias = self.bias
            outdims = self.outdims
        else:
            if isinstance(out_idx, np.ndarray):
                if out_idx.dtype == bool:
                    out_idx = np.where(out_idx)[0]
            feat = feat[:, :, :, out_idx]
            grid = self.grid[:, out_idx]
            if self.bias is not None:
                bias = self.bias[out_idx]
            outdims = len(out_idx)

        if shift is None:
            grid = grid.expand(N, outdims, 1, 2)
        else:
            # shift grid based on shifter network's prediction
            grid = grid.expand(N, outdims, 1, 2) + shift[:, None, None, :]

        pools = [F.grid_sample(x, grid, align_corners=self.align_corners)]
        for _ in range(self.pool_steps):
            _, _, w_pool, h_pool = x.size()
            if w_pool * h_pool == 1:
                warnings.warn("redundant pooling steps: pooled feature map size is already 1X1, consider reducing it")
            x = self.avg(x)
            pools.append(F.grid_sample(x, grid, align_corners=self.align_corners))
        y = torch.cat(pools, dim=1)
        y = (y.squeeze(-1).unsqueeze(0) * feat).sum(2).view(self.inferred_params_n, N, outdims)

        if self.bias is not None:
            y = y + bias.unsqueeze(1)
        return y.squeeze()
