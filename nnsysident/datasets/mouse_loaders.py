from collections import OrderedDict
from itertools import zip_longest

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from neuralpredictors.data.datasets import FileTreeDataset, StaticImageSet
from neuralpredictors.data.samplers import SubsetSequentialSampler
from neuralpredictors.data.transforms import AddBehaviorAsChannels, NeuroNormalizer, Subsample, ToTensor

from ..utility.data_helpers import extract_data_key, get_oracle_dataloader, set_random_seed
from .transforms import filter_neurons, get_transforms

try:
    from dataport.bcm.static import fetch_non_existing_data
except ImportError:

    def fetch_non_existing_data(func):
        return func

    print("dataport not available, will only be able to load data locally")


@fetch_non_existing_data
def static_loader(
    path,
    batch_size,
    areas=None,
    layers=None,
    tier=None,
    neuron_ids=None,
    neuron_n=None,
    exclude_neuron_n=0,
    neuron_base_seed=None,
    image_ids=None,
    image_n=None,
    image_base_seed=None,
    trial_indices=None,
    get_key=False,
    cuda=True,
    normalize=True,
    exclude="images",
    loader_outputs=["images", "responses"],
    select_input_channel=None,
    file_tree=True,
    return_test_sampler=False,
    oracle_condition=None,
    shuffle_train=True,
    shuffle_test=False,
    scale=None,
    subtract_behavior_mean=False,
):
    """
    returns a single data loader

    Args:
        path (str): path for the dataset
        batch_size (int): batch size.
        areas (list, optional): the visual area.
        layers (list, optional): the layer from visual area.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader.
        neuron_ids (list, optional): select neurons by their ids.
        neuron_n (int, optional): number of neurons to select randomly. Can not be set together with neuron_ids
        neuron_base_seed (float, optional): base seed for neuron selection. Gets multiplied by neuron_n to obtain final seed
        exclude_neuron_n (int): the first <exclude_neuron_n> neurons will be excluded (given a neuron_base_seed),
                                then <neuron_n> neurons will be drawn from the remaining neurons.
        image_ids (list, optional): select images by their ids.
        image_n (int, optional): number of images to select randomly. Can not be set together with image_ids
        image_base_seed (float, optional): base seed for image selection. Gets multiplied by image_n to obtain final seed
        trial_indices (list, optional): select trials by their ids.
        get_key (bool, optional): whether to return the data key, along with the dataloaders.
        cuda (bool, optional): whether to place the data on gpu or not.
        normalize (bool, optional): whether to normalize the data (see also exclude)
        exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
        loader_outputs (list): list of data that the loader should give as an output. Defaults to ["images", "responses"]
        select_input_channel (int, optional): Only for color images. Select a color channel
        file_tree (bool, optional): whether to use the file tree dataset format. If False, equivalent to the HDF5 format
        return_test_sampler (bool, optional): whether to return only the test loader with repeat-batches
        oracle_condition (list, optional): Only relevant if return_test_sampler=True. Class indices for the sampler
        shuffle_train (bool, optional): whether to shuffle the train set
        shuffle_test (bool, optional): whether to shuffle the test set
        scale (float, optional): scale the image size
        subtract_behavior_mean (bool, optional): whether to mean-normalize the behavior variables (if available).

    Returns:
        if get_key is False returns a dictionary of dataloaders for one dataset, where the keys are 'train', 'validation', and 'test'.
        if get_key is True it returns the data_key (as the first output) followed by the dataloder dictionary.

    """
    assert (image_ids is not None) + (image_n is not None) + (
        trial_indices is not None
    ) <= 1, "Only one out of {image_ids, image_n, trial_indices} can be set."

    if image_n is None and image_base_seed is not None:
        warn("image_base_seed is going to be ignored because image_n is set to None.")

    data_key = extract_data_key(path)

    if file_tree:
        dat = FileTreeDataset(path, *loader_outputs)
    else:
        dat = StaticImageSet(path, *loader_outputs)

    idx = filter_neurons(dat, neuron_ids, neuron_n, neuron_base_seed, areas, layers, exclude_neuron_n)
    transforms = get_transforms(
        dat, idx, normalize, exclude, loader_outputs, select_input_channel, scale, cuda, subtract_behavior_mean
    )
    dat.transforms.extend(transforms)

    if return_test_sampler:
        print("Returning only test sampler with repeats...")
        dataloader = get_oracle_dataloader(dat, oracle_condition=oracle_condition, file_tree=file_tree)
        return (data_key, {"test": dataloader}) if get_key else {"test": dataloader}

    # subsample images
    dataloaders = {}
    keys = [tier] if tier else ["train", "validation", "test"]
    shuffle = {"train": shuffle_train, "validation": False, "test": shuffle_test}
    tier_array = dat.trial_info.tiers if file_tree else dat.tiers


    for tier in keys:
        # sample images
        if image_ids is not None:
            image_id_array = dat.trial_info.frame_image_id if file_tree else dat.info.frame_image_id
            image_ids_per_tier = list(set(image_ids) & set(image_id_array[tier_array == tier]))
            subset_idx = np.where(np.isin(image_id_array, image_ids_per_tier))[0]

        elif tier == "train" and image_n is not None:
            random_state = np.random.get_state()
            if image_base_seed is not None:
                np.random.seed(image_base_seed * image_n)  # avoid nesting by making seed dependent on number of images
            subset_idx = np.random.choice(np.where(tier_array == "train")[0], size=image_n, replace=False)
            np.random.set_state(random_state)

        elif trial_indices is not None:
            tier_trial_indices = np.array(trial_indices)[
                np.isin(trial_indices, dat.trial_info.trial_idx[tier_array == tier])
            ]
            subset_idx = np.where(np.isin(dat.trial_info.trial_idx, tier_trial_indices))[0]

        else:
            subset_idx = np.where(tier_array == tier)[0]

        sampler = SubsetRandomSampler(subset_idx) if shuffle[tier] else SubsetSequentialSampler(subset_idx)
        dataloaders[tier] = DataLoader(dat, sampler=sampler, batch_size=batch_size)

    # create the data_key for a specific data path
    return (data_key, dataloaders) if get_key else dataloaders


def static_loaders(
    paths, batch_size, seed=None, tier=None, neuron_ids=None, image_ids=None, trial_indices=None, **kwargs
):
    """
    Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).

    Args:
        paths (list): list of paths for the datasets
        batch_size (int): batch size.
        seed (int): seed. Not really needed because there are neuron and image seed. But nnFabrik requires it.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader.
        neuron_ids (list, optional): List of lists of neuron_ids. Make sure the order is the same as in paths
        image_ids (list, optional): List of lists of image_ids. Make sure the order is the same as in paths
        trial_indices (list, optional): List of lists of trial_indices. Make sure the order is the same as in paths

        For all other arguments, see static_loader

    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """
    set_random_seed(seed)
    dls = OrderedDict({})
    keys = [tier] if tier else ["train", "validation", "test"]
    for key in keys:
        dls[key] = OrderedDict({})

    neuron_ids = [neuron_ids] if neuron_ids is None else neuron_ids
    image_ids = [image_ids] if image_ids is None else image_ids
    trial_indices = [trial_indices] if trial_indices is None else trial_indices

    for path, neuron_id, image_id, trial_index in zip_longest(
        paths, neuron_ids, image_ids, trial_indices, fillvalue=None
    ):
        data_key, loaders = static_loader(
            path,
            batch_size,
            tier=tier,
            get_key=True,
            neuron_ids=neuron_id,
            image_ids=image_id,
            trial_indices=trial_index,
            **kwargs,
        )
        for k in dls:
            dls[k][data_key] = loaders[k]

    return dls


def static_shared_loaders(
    paths,
    batch_size,
    seed=None,
    tier=None,
    image_ids=None,
    multi_match_ids=None,
    multi_match_n=None,
    exclude_multi_match_n=0,
    multi_match_base_seed=None,
    **kwargs,
):
    """
    Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).
    The datasets must have matched neurons. Only the file tree format is supported.

    Args:
        paths (list): list of paths for the datasets
        batch_size (int): batch size.
        seed (int): seed. Not really needed because there are neuron and image seed. But nnFabrik requires it.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader.
        multi_match_ids (list, optional): List of multi_match_ids according to which the respective neuron_ids are drawn for each dataset in paths
        multi_match_n (int, optional): number of neurons to select randomly. Can not be set together with multi_match_ids
        exclude_multi_match_n (int): the first <exclude_multi_match_n> matched neurons will be excluded (given a multi_match_base_seed),
                                then <multi_match_n> matched neurons will be drawn from the remaining neurons.
        multi_match_base_seed (float, optional): base seed for neuron selection. Get's multiplied by multi_match_n to obtain final seed
        image_ids (list, optional): List of lists of image_ids. Make sure the order is the same as in paths

        For all other arguments, see static_loader

    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """
    if seed is not None:
        set_random_seed(seed)
    assert (
        len(paths) != 1
    ), "Only one dataset was specified in 'paths'. When using the 'static_shared_loaders', more than one dataset has to be passed."
    assert any(
        [
            multi_match_ids is None,
            all([multi_match_n is None, multi_match_base_seed is None, exclude_multi_match_n == 0]),
        ]
    ), "multi_match_ids can not be set at the same time with any other multi_match selection criteria"
    assert any(
        [exclude_multi_match_n == 0, multi_match_base_seed is not None]
    ), "multi_match_base_seed must be set when exclude_multi_match_n is not 0"
    # Collect overlapping multi matches
    multi_unit_ids, per_data_set_ids, given_neuron_ids = [], [], []
    match_set = None
    for path in paths:
        data_key, dataloaders = static_loader(path=path, batch_size=100, get_key=True)
        dat = dataloaders["train"].dataset
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

    # get unit_ids of matched neurons
    if multi_match_ids is not None:
        neuron_ids = given_neuron_ids
    elif multi_match_n is not None:
        random_state = np.random.get_state()
        if multi_match_base_seed is not None:
            np.random.seed(
                multi_match_base_seed * multi_match_n
            )  # avoid nesting by making seed dependent on number of neurons
        assert (
            len(match_set) >= exclude_multi_match_n + multi_match_n
        ), "After excluding {} neurons, there are not {} matched neurons left".format(
            exclude_multi_match_n, multi_match_n
        )
        match_subset = np.random.choice(match_set, size=exclude_multi_match_n + multi_match_n, replace=False)[
            exclude_multi_match_n:
        ]
        neuron_ids = [
            pdsi[np.isin(munit_ids, match_subset)] for munit_ids, pdsi in zip(multi_unit_ids, per_data_set_ids)
        ]
        np.random.set_state(random_state)
    else:
        neuron_ids = [pdsi[np.isin(munit_ids, match_set)] for munit_ids, pdsi in zip(multi_unit_ids, per_data_set_ids)]

    # generate single dataloaders
    dls = OrderedDict({})
    keys = [tier] if tier else ["train", "validation", "test"]
    for key in keys:
        dls[key] = OrderedDict({})

    image_ids = [image_ids] if image_ids is None else image_ids
    for path, neuron_id, image_id in zip_longest(paths, neuron_ids, image_ids, fillvalue=None):

        data_key, loaders = static_loader(
            path, batch_size, tier=tier, get_key=True, neuron_ids=neuron_id, image_ids=image_id, **kwargs
        )
        for k in dls:
            dls[k][data_key] = loaders[k]

    return dls


def mouse_allen_scene_loader(
    path=None,
    batch_size=None,
    seed=None,
    areas=None,
    imaging_depths=None,
    tier=None,
    specimen_ids=None,
    get_key=False,
    cuda=True,
    normalize=True,
    include_behavior=False,
    exclude="images",
    return_test_sampler=False,
    neuron_ids=None,
    select_input_channel=None,
    oracle_condition=None,
):
    print(neuron_ids)
    assert neuron_ids is None, "neuron_ids not implemented yet"
    assert select_input_channel is None, "select_input_channel not implemented yet"

    dat = (
        FileTreeDataset(path, "images", "responses", "behavior")
        if include_behavior
        else FileTreeDataset(path, "images", "responses")
    )
    # specify condition(s) for sampling neurons. If you want to sample specific neurons define conditions that would effect idx
    conds = np.ones(len(dat.neurons.area), dtype=bool)
    if areas is not None:
        conds &= np.isin(dat.neurons.area, areas)
    if imaging_depths is not None:
        conds &= np.isin(dat.neurons.imaging_depth, imaging_depths)
    if specimen_ids is not None:
        conds &= np.isin(dat.neurons.specimen_ids, specimen_ids)
    idx = np.where(conds)[0]
    more_transforms = [Subsample(idx), ToTensor(cuda)]
    if normalize:
        more_transforms.insert(0, NeuroNormalizer(dat, exclude=exclude))
    if include_behavior:
        more_transforms.insert(0, AddBehaviorAsChannels())
    dat.transforms.extend(more_transforms)
    if return_test_sampler:
        assert False, "Check that code before you run it"
        dataloader = get_oracle_dataloader(dat, oracle_condition=oracle_condition, file_tree=True)
        return dataloader
    # subsample images
    dataloaders = {}
    keys = [tier] if tier else ["train", "validation", "test"]
    for tier in keys:
        if seed is not None:
            set_random_seed(seed)
        # sample images
        subset_idx = np.where(dat.trial_info.tiers == tier)[0]
        sampler = SubsetRandomSampler(subset_idx) if tier == "train" else SubsetSequentialSampler(subset_idx)
        dataloaders[tier] = DataLoader(dat, sampler=sampler, batch_size=batch_size)
    # create the data_key for a specific data path
    data_key = path.split("allen")[-1].split(".")[0]
    return (data_key, dataloaders) if get_key else dataloaders


def mouse_allen_scene_loaders(
    paths,
    batch_size,
    seed=None,
    areas=None,
    imaging_depths=None,
    tier=None,
    specimen_ids=None,
    cuda=True,
    normalize=True,
    include_behavior=False,
    exclude="images",
    select_input_channel=None,
):
    """
    Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).
    Args:
        paths (list): list of path(s) for the dataset(s)
        batch_size (int): batch size.
        seed (int, optional): random seed for images. Defaults to None.
        areas (str, optional): the visual area. Defaults to 'V1'.
        layers (str, optional): the layer from visual area. Defaults to 'L2/3'.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader. Defaults to None.
        neuron_ids ([type], optional): select neurons by their ids. Defaults to None.
        cuda (bool, optional): whether to place the data on gpu or not. Defaults to True.
    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """
    neuron_ids = specimen_ids if specimen_ids is not None else [None]
    dls = OrderedDict({})
    keys = [tier] if tier else ["train", "validation", "test"]
    for key in keys:
        dls[key] = OrderedDict({})
    for path, neuron_id in zip_longest(paths, neuron_ids, fillvalue=None):
        data_key, loaders = mouse_allen_scene_loader(
            path,
            batch_size,
            seed=seed,
            areas=areas,
            imaging_depths=imaging_depths,
            cuda=cuda,
            tier=tier,
            get_key=True,
            neuron_ids=neuron_id,
            normalize=normalize,
            include_behavior=include_behavior,
            exclude=exclude,
            select_input_channel=select_input_channel,
        )
        for k in dls:
            dls[k][data_key] = loaders[k]
    return dls
