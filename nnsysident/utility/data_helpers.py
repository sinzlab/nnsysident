import random
import re
from collections import Counter

import numpy as np
import torch
import torch.utils.data as utils

from neuralpredictors.data.samplers import RepeatsBatchSampler


def get_oracle_dataloader(
    dat, toy_data=False, oracle_condition=None, verbose=False, file_tree=False, subset_idx=None, min_count=2
):

    if toy_data:
        condition_hashes = dat.info.condition_hash
        image_class = "toy_data"
    else:
        dat_info = dat.info if not file_tree else dat.trial_info
        if "image_id" in dir(dat_info):
            condition_hashes = dat_info.image_id
            image_class = dat_info.image_class

        elif "colorframeprojector_image_id" in dir(dat_info):
            condition_hashes = dat_info.colorframeprojector_image_id
            image_class = dat_info.colorframeprojector_image_class
        elif "frame_image_id" in dir(dat_info):
            condition_hashes = dat_info.frame_image_id
            image_class = dat_info.frame_image_class
        else:
            raise ValueError(
                "'image_id' 'colorframeprojector_image_id', or 'frame_image_id' have to present in the dataset under dat.info "
                "in order to load get the oracle repeats."
            )

    max_idx = condition_hashes.max() + 1
    classes, class_idx = np.unique(image_class, return_inverse=True)
    identifiers = condition_hashes + class_idx * max_idx

    min_count_elements = [k for k, v in Counter(condition_hashes).items() if v >= min_count]
    min_count_condition = np.isin(condition_hashes, min_count_elements)

    sampling_condition = (
        np.where(min_count_condition)[0]
        if oracle_condition is None
        else np.where((min_count_condition) & (class_idx == oracle_condition))[0]
    )
    if (oracle_condition is not None) and verbose:
        print("Created Testloader for image class {}".format(classes[oracle_condition]))
    if subset_idx is not None:
        sampling_condition = np.array(list(set(subset_idx) & set(sampling_condition)))
    sampler = RepeatsBatchSampler(identifiers, sampling_condition)
    return utils.DataLoader(dat, batch_sampler=sampler)


def unpack_data_info(data_info):

    in_shapes_dict = {k: v["input_dimensions"] for k, v in data_info.items()}
    input_channels = [v["input_channels"] for k, v in data_info.items()]
    n_neurons_dict = {k: v["output_dimension"] for k, v in data_info.items()}
    return n_neurons_dict, in_shapes_dict, input_channels


def extract_data_key(path):
    return "-".join((re.findall(r"\d+", path)[:3] + ["0"]))


def get_io_dims(data_loader):
    """
    --- Copied from nnfabrik ---

    Returns the shape of the dataset for each item within an entry returned by the `data_loader`
    The DataLoader object must return either a namedtuple, dictionary or a plain tuple.
    If `data_loader` entry is a namedtuple or a dictionary, a dictionary with the same keys as the
    namedtuple/dict item is returned, where values are the shape of the entry. Otherwise, a tuple of
    shape information is returned.
    Note that the first dimension is always the batch dim with size depending on the data_loader configuration.
    Args:
        data_loader (torch.DataLoader): is expected to be a pytorch Dataloader object returning
            either a namedtuple, dictionary, or a plain tuple.
    Returns:
        dict or tuple: If data_loader element is either namedtuple or dictionary, a ditionary
            of shape information, keyed for each entry of dataset is returned. Otherwise, a tuple
            of shape information is returned. The first dimension is always the batch dim
            with size depending on the data_loader configuration.
    """
    items = next(iter(data_loader))
    if hasattr(items, "_asdict"):  # if it's a named tuple
        items = items._asdict()

    if hasattr(items, "items"):  # if dict like
        return {k: v.shape for k, v in items.items()}
    else:
        return (v.shape for v in items)


def get_mean_activity_dict(dataloaders):
    # initializing readout bias to mean response
    mean_activity_dict = {}
    for key, value in dataloaders.items():
        data = next(iter(value))
        if "targets" in data._fields:
            mean_activity_dict[key] = data.targets.mean(0)
        elif "responses" in data._fields:
            mean_activity_dict[key] = data.responses.mean(0)
        else:
            raise ValueError()
    return mean_activity_dict


def get_dims_for_loader_dict(dataloaders):
    """
    --- Copied from nnfabrik ---

    Given a dictionary of DataLoaders, returns a dictionary with same keys as the
    input and shape information (as returned by `get_io_dims`) on each keyed DataLoader.
    Args:
        dataloaders (dict of DataLoader): Dictionary of dataloaders.
    Returns:
        dict: A dict containing the result of calling `get_io_dims` for each entry of the input dict
    """
    return {k: get_io_dims(v) for k, v in dataloaders.items()}


def set_random_seed(seed: int, deterministic: bool = True):
    """
    --- Copied from nnfabrik ---

    Set random generator seed for Python interpreter, NumPy and PyTorch. When setting the seed for PyTorch,
    if CUDA device is available, manual seed for CUDA will also be set. Finally, if `deterministic=True`,
    and CUDA device is available, PyTorch CUDNN backend will be configured to `benchmark=False` and `deterministic=True`
    to yield as deterministic result as possible. For more details, refer to
    PyTorch documentation on reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    Beware that the seed setting is a "best effort" towards deterministic run. However, as detailed in the above documentation,
    there are certain PyTorch CUDA opertaions that are inherently non-deterministic, and there is no simple way to control for them.
    Thus, it is best to assume that when CUDA is utilized, operation of the PyTorch module will not be deterministic and thus
    not completely reproducible.
    Args:
        seed (int): seed value to be set
        deterministic (bool, optional): If True, CUDNN backend (if available) is set to be deterministic. Defaults to True. Note that if set
            to False, the CUDNN properties remain untouched and it NOT explicitly set to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)  # this sets both CPU and CUDA seeds for PyTorch
