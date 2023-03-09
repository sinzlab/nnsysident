import numpy as np
from neuralpredictors.data.transforms import (
    Subsample,
    ToTensor,
    NeuroNormalizer,
    SelectInputChannel,
    ScaleInputs,
    StaticTransform
)

class NoNegativeResponses(StaticTransform):
    def __call__(self, x):
        x[1][x[1] < 0.0] = 0.0
        return x

def filter_neurons(dat, neuron_ids, neuron_n, neuron_base_seed, areas, layers, exclude_neuron_n):
    assert any(
        [
            neuron_ids is None,
            all(
                [
                    neuron_n is None,
                    neuron_base_seed is None,
                    areas is None,
                    layers is None,
                    exclude_neuron_n == 0,
                    ]
            ),
            ]
    ), "neuron_ids can not be set at the same time with any other neuron selection criteria"

    assert any(
        [exclude_neuron_n == 0, neuron_base_seed is not None]
    ), "neuron_base_seed must be set when exclude_neuron_n is not 0"

    # The permutation MUST be added first and the conditions below MUST NOT be based on the original order
    # specify condition(s) for sampling neurons. If you want to sample specific neurons define conditions that would effect idx
    conds = np.ones(len(dat.neurons.area), dtype=bool)
    if areas is not None:
        conds &= np.isin(dat.neurons.area, areas)
    if layers is not None:
        conds &= np.isin(dat.neurons.layer, layers)
    idx = np.where(conds)[0]
    if neuron_n is not None:
        random_state = np.random.get_state()
        if neuron_base_seed is not None:
            np.random.seed(neuron_base_seed * neuron_n)  # avoid nesting by making seed dependent on number of neurons
        assert (
                len(dat.neurons.unit_ids) >= exclude_neuron_n + neuron_n
        ), "After excluding {} neurons, there are not {} neurons left".format(exclude_neuron_n, neuron_n)
        neuron_ids = np.random.choice(dat.neurons.unit_ids, size=exclude_neuron_n + neuron_n, replace=False)[
                     exclude_neuron_n:
                     ]
        np.random.set_state(random_state)
    if neuron_ids is not None:
        idx = [np.where(dat.neurons.unit_ids == unit_id)[0][0] for unit_id in neuron_ids]

    return idx

def get_transforms(dat, idx, normalize, exclude, loader_outputs, select_input_channel, scale, cuda, subtract_behavior_mean=False):
    assert not (
            "behavior" in loader_outputs and select_input_channel is not None
    ), "Selecting an Input Channel and Adding Behavior can not both be true"
    transforms = [Subsample(idx), NoNegativeResponses(), ToTensor(cuda)]

    if scale is not None:
        transforms.insert(0, ScaleInputs(scale=scale))

    if normalize:
        transforms.insert(0, NeuroNormalizer(dat, exclude=exclude, subtract_behavior_mean=subtract_behavior_mean))

    if select_input_channel is not None:
        transforms.insert(0, SelectInputChannel(select_input_channel))

    return transforms
