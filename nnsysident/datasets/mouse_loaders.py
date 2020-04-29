from collections import OrderedDict
from itertools import zip_longest
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from mlutils.data.datasets import StaticImageSet, FileTreeDataset
from mlutils.data.transforms import Subsample, ToTensor, NeuroNormalizer, AddBehaviorAsChannels, SelectInputChannel
from mlutils.data.samplers import SubsetSequentialSampler
from nnfabrik.utility.nn_helpers import set_random_seed
from .utility import get_oracle_dataloader



def static_loader(
    path,
    batch_size,
    areas=None,
    layers=None,
    tier=None,
    neuron_ids=None,
    image_ids=None,
    get_key=False,
    cuda=True,
    normalize=True,
    exclude="images",
    include_behavior=False,
    select_input_channel=None,
    toy_data=False,
    file_tree=True,
    return_test_sampler=False,
    oracle_condition=None,
    **kwargs
):
    """
    returns a single data loader

    Args:
        path (list): list of path(s) for the dataset(s)
        batch_size (int): batch size.
        areas (list, optional): the visual area.
        layers (list, optional): the layer from visual area.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader.
        neuron_ids (list, optional): select neurons by their ids. neuron_ids and path should be of same length.
        image_ids (list, optional): select images by their ids. image_ids and path should be of same length.
        get_key (bool, optional): whether to return the data key, along with the dataloaders.
        cuda (bool, optional): whether to place the data on gpu or not.
        normalize (bool, optional): whether to normalize the data (see also exclude)
        exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
        include_behavior (bool, optional): whether to include behavioral data
        select_input_channel (int, optional): Only for color images. Select a color channel
        toy_data: (bool, optional): whether to use data as toy data
        file_tree (bool, optional): whether to use the file tree dataset format. If False, equivalent to the HDF5 format
        return_test_sampler (bool, optional): whether to return only the test loader with repeat-batches
        oracle_condition (list, optional): Only relevant if return_test_sampler=True. Class indices for the sampler

    Returns:
        if get_key is False returns a dictionary of dataloaders for one dataset, where the keys are 'train', 'validation', and 'test'.
        if get_key is True it returns the data_key (as the first output) followed by the dataloder dictionary.

    """
    if ("paths" in kwargs) and (path is None):
        paths = kwargs["paths"]
        path = paths[0] if len(paths) == 1 else None

    if file_tree:
        dat = (
            FileTreeDataset(path, "images", "responses", "behavior")
            if include_behavior
            else FileTreeDataset(path, "images", "responses")
        )
    else:
        dat = (
            StaticImageSet(path, "images", "responses", "behavior")
            if include_behavior
            else StaticImageSet(path, "images", "responses")
        )

    assert (
        include_behavior and select_input_channel
    ) is False, "Selecting an Input Channel and Adding Behavior can not both be true"

    if toy_data:
        dat.transforms = [ToTensor(cuda)]
    else:
        # The permutation MUST be added first and the conditions below MUST NOT be based on the original order

        # specify condition(s) for sampling neurons. If you want to sample specific neurons define conditions that would effect idx
        conds = np.ones(len(dat.neurons.area), dtype=bool)
        if areas is not None:
            conds &= np.isin(dat.neurons.area, areas)
        if layers is not None:
            conds &= np.isin(dat.neurons.layer, layers)
        idx = np.where(conds)[0]
        if neuron_ids is not None:
            assert all(
                conds
            ), "If neuron_ids are given, no other neuron selection criteria like area or layer may be chosen"
            idx = [np.where(dat.neurons.unit_ids == unit_id)[0][0] for unit_id in neuron_ids]

        more_transforms = [Subsample(idx), ToTensor(cuda)]
        if normalize:
            more_transforms.insert(0, NeuroNormalizer(dat, exclude=exclude))

        if include_behavior:
            more_transforms.insert(0, AddBehaviorAsChannels())

        if select_input_channel is not None:
            more_transforms.insert(0, SelectInputChannel(select_input_channel))

        dat.transforms.extend(more_transforms)

    if return_test_sampler:
        dataloader = get_oracle_dataloader(
            dat, toy_data=toy_data, oracle_condition=oracle_condition, file_tree=file_tree
        )
        return dataloader

    # subsample images
    dataloaders = {}
    keys = [tier] if tier else ["train", "validation", "test"]
    tier_array = dat.trial_info.tiers if file_tree else dat.tiers
    image_id_array = dat.trial_info.frame_image_id if file_tree else dat.info.frame_image_id
    for tier in keys:
        # sample images
        if tier == "train" and image_ids is not None:
            subset_idx = [np.where(image_id_array == image_id)[0][0] for image_id in image_ids]
            assert sum(tier_array[subset_idx] != 'train') == 0, 'image_ids contain validation or test images'
        else:
            subset_idx = np.where(tier_array == tier)[0]

        sampler = SubsetRandomSampler(subset_idx) if tier == 'train' else SubsetSequentialSampler(subset_idx)
        dataloaders[tier] = DataLoader(dat, sampler=sampler, batch_size=batch_size)

    # create the data_key for a specific data path
    data_key = path.split("static")[-1].split(".")[0].replace("preproc", "").replace("_nobehavior", "")
    return (data_key, dataloaders) if get_key else dataloaders


def static_loaders(
    paths,
    batch_size,
    areas=None,
    layers=None,
    tier=None,
    neuron_ids=None,
    image_ids=None,
    cuda=True,
    normalize=True,
    include_behavior=False,
    exclude="images",
    select_input_channel=None,
    toy_data=False,
    file_tree=True,
    **kwargs
):
    """
    Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).

    Args:
        paths (list): list of paths for the datasets
        batch_size (int): batch size.
        areas (list, optional): the visual area.
        layers (list, optional): the layer from visual area.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader.
        neuron_ids (list, optional): select neurons by their ids. neuron_ids and path should be of same length.
        image_ids (list, optional): select images by their ids. image_ids and path should be of same length.
        cuda (bool, optional): whether to place the data on gpu or not.
        normalize (bool, optional): whether to normalize the data (see also exclude)
        exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
        include_behavior (bool, optional): whether to include behavioral data
        select_input_channel (int, optional): Only for color images. Select a color channel
        toy_data: (bool, optional): whether to use data as toy data
        file_tree (bool, optional): whether to use the file tree dataset format. If False, equivalent to the HDF5 format

    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """

    dls = OrderedDict({})
    keys = [tier] if tier else ["train", "validation", "test"]
    for key in keys:
        dls[key] = OrderedDict({})

    for path, neuron_id in zip_longest(paths, neuron_ids, fillvalue=None):
        data_key, loaders = static_loader(
            path,
            batch_size,
            seed=seed,
            areas=areas,
            layers=layers,
            cuda=cuda,
            tier=tier,
            get_key=True,
            neuron_ids=neuron_id,
            image_ids=image_ids,
            normalize=normalize,
            include_behavior=include_behavior,
            exclude=exclude,
            select_input_channel=select_input_channel,
            toy_data=toy_data,
            file_tree=file_tree,
        )
        for k in dls:
            dls[k][data_key] = loaders[k]

    return dls


def static_shared_loaders(
    paths,
    batch_size,
    areas=None,
    layers=None,
    tier=None,
    multi_match_ids=None,
    image_ids=None,
    cuda=True,
    normalize=True,
    include_behavior=False,
    exclude="images",
    select_input_channel=None,
    toy_data=False,
    **kwargs
):
    """
    Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).
    The datasets must have matched neurons. Only the file tree format is supported.

    Args:
        paths (list): list of paths for the datasets
        batch_size (int): batch size.
        areas (list, optional): the visual area.
        layers (list, optional): the layer from visual area.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader.
        multi_match_ids (list, optional): select neurons by their ids. neuron_ids and path should be of same length.
        image_ids (list, optional): select images by their ids. image_ids and path should be of same length.
        cuda (bool, optional): whether to place the data on gpu or not.
        normalize (bool, optional): whether to normalize the data (see also exclude)
        exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
        include_behavior (bool, optional): whether to include behavioral data
        select_input_channel (int, optional): Only for color images. Select a color channel
        toy_data: (bool, optional): whether to use data as toy data

    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """

    assert (
        len(paths) != 1
    ), "Only one dataset was specified in 'paths'. When using the 'static_shared_loaders', more than one dataset has to be passed."

    # Collect overlapping multi matches
    multi_unit_ids, per_data_set_ids, given_neuron_ids = [], [], []
    match_set = None
    for path in paths:
        dat = FileTreeDataset(path, "responses")
        multi_unit_ids.append(dat.neurons.multi_match_id)
        per_data_set_ids.append(dat.neurons.unit_ids)
        if match_set is None:
            match_set = set(multi_unit_ids[-1])
        else:
            match_set &= set(multi_unit_ids[-1])
        if multi_match_ids is not None:
            assert set(multi_match_ids).issubset(
                dat.neurons.multi_match_id
            ), "Dataset {} does not contain all multi_match_ids".format(path)
            neuron_idx = [
                np.where(dat.neurons.multi_match_id == multi_match_id)[0][0] for multi_match_id in multi_match_ids
            ]
            given_neuron_ids.append(dat.neurons.unit_ids[neuron_idx])
    match_set -= {-1}  # remove the unmatched neurons
    match_set = np.array(list(match_set))

    # get unit_ids of intersecting multi-unit ids
    all_set_neurons = [pdsi[np.isin(munit_ids, match_set)] for munit_ids, pdsi in zip(multi_unit_ids, per_data_set_ids)]
    neuron_ids = all_set_neurons if multi_match_ids is None else given_neuron_ids

    # generate single dataloaders
    dls = OrderedDict({})
    keys = [tier] if tier else ["train", "validation", "test"]
    for key in keys:
        dls[key] = OrderedDict({})

    for path, neuron_id in zip(paths, neuron_ids):
        data_key, loaders = static_loader(
            path,
            batch_size,
            seed=seed,
            areas=areas,
            layers=layers,
            cuda=cuda,
            tier=tier,
            get_key=True,
            neuron_ids=neuron_id,
            image_ids=image_ids,
            normalize=normalize,
            include_behavior=include_behavior,
            exclude=exclude,
            select_input_channel=select_input_channel,
            toy_data=toy_data,
            file_tree=True,
        )
        for k in dls:
            dls[k][data_key] = loaders[k]

    return dls
