import re
import numpy as np
import torch.utils.data as utils
from collections import Counter

from neuralpredictors.data.samplers import RepeatsBatchSampler


def get_oracle_dataloader(dat,
                          toy_data=False,
                          oracle_condition=None,
                          verbose=False,
                          file_tree=False,
                          subset_idx=None,
                          min_count=2):

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