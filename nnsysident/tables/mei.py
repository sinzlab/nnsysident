from __future__ import annotations

import os
import shutil
import uuid
from typing import Any, Dict
from tqdm import tqdm

import datajoint as dj
import torch
import numpy as np
from nnfabrik.main import Dataset, my_nnfabrik
from nnvision.tables.main import Recording
from torch import load
from torch.nn import Module
from torch.utils.data import DataLoader
from .experiments import Experiments

if not "stores" in dj.config:
    dj.config["stores"] = {}
dj.config["stores"]["minio"] = {
    "protocol": "s3",
    "endpoint": os.environ["MINIO_ENDPOINT"],
    "bucket": "kklurzmodels",
    "location": "dj-store",
    "access_key": os.environ["MINIO_ACCESS_KEY"],
    "secret_key": os.environ["MINIO_SECRET_KEY"],
    "secure": True,
}

from mei.main import (
    CSRFV1ObjectiveTemplate,
    MEIMethod,
    MEISeed,
    MEITemplate,
    TrainedEnsembleModelTemplate,
)
from mei.modules import ConstrainedOutputModel

from .experiments import TrainedModel

Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]

# create the context object
try:
    main = my_nnfabrik(os.environ["DJ_SCHEMA_NAME"], use_common_fabrikant=False)
except:
    raise ValueError(
        " ".join(
            [
                "No schema name has been specified.",
                "Specify it via",
                "os.environ['DJ_SCHEMA_NAME']='schema_name'",
            ]
        )
    )
# set some local variables such that the tables can be directly importable elsewhere
for key, val in main.__dict__.items():
    locals()[key] = val


class MouseSelectorTemplate(dj.Computed):

    dataset_table = Dataset
    definition = """
    # contains information that can be used to map a neuron's id to its corresponding integer position in the output of
    # the model. 
    -> self.dataset_table
    unit_id       : int               # unique neuron identifier
    data_key        : varchar(255)      # unique session identifier
    ---
    unit_index : int                    # integer position of the neuron in the model's output 
    """

    constrained_output_model = ConstrainedOutputModel

    def make(self, key):
        dataloaders = (Dataset & key).get_dataloader()
        mappings = []
        for data_key, loader in dataloaders["train"].items():
            neuron_ids = loader.dataset.neurons.unit_ids
            for neuron_pos, neuron_id in enumerate(neuron_ids):
                mappings.append(dict(key, unit_id=neuron_id, unit_index=neuron_pos, data_key=data_key))

        self.insert(mappings)

    def get_output_selected_model(self, model: Module, key: Key) -> constrained_output_model:
        unit_index, data_key = (self & key).fetch1("unit_index", "data_key")
        return self.constrained_output_model(model, unit_index, forward_kwargs=dict(data_key=data_key))


@schema
class MEISelector(MouseSelectorTemplate):
    dataset_table = Dataset


@schema
class TrainedEnsembleModel(TrainedEnsembleModelTemplate):
    dataset_table = Dataset
    trained_model_table = TrainedModel


@schema
class MEI(MEITemplate):
    trained_model_table = TrainedEnsembleModel
    selector_table = MEISelector

    def load_data(self, names, numpy=True):
        download_path = '/project/notebooks/data_' + str(uuid.uuid4())
        data = (self * self.method_table).fetch(*names, download_path=download_path)

        if "mei" in names:
            idx = names.index("mei")
            meis = []
            for path in tqdm(data[idx], total=len(data[idx])):
                mei = load(path).data.numpy() if numpy else load(path)
                meis.append(mei)
            data[idx] = np.stack(meis) if numpy else torch.stack(meis)
            shutil.rmtree(download_path)
        return data



@schema
class MEIMonkey(MEITemplate):
    trained_model_table = TrainedEnsembleModel
    selector_table = Recording.Units


@schema
class MEIScore(dj.Computed):
    """
    A template for a MEI scoring table.
    """

    mei_table = MEI
    measure_attribute = "score"
    function_kwargs = {}
    external_download_path = None

    # table level comment
    table_comment = "A template table for storing results/scores of a MEI"

    @property
    def definition(self):
        definition = """
                    # {table_comment}
                    -> self.mei_table
                    ---
                    {measure_attribute}:      float     # A template for a computed score of a trained model
                    {measure_attribute}_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
                    """.format(
            table_comment=self.table_comment, measure_attribute=self.measure_attribute
        )
        return definition

    @staticmethod
    def measure_function(mei, **kwargs):
        raise NotImplementedError("Scoring Function has to be implemented")

    def get_mei(self, key):
        mei = torch.load((self.mei_table & key).fetch1("mei", download_path=self.external_download_path))
        return mei

    def make(self, key):
        mei = self.get_mei(key=key)
        score = self.measure_function(mei, **self.function_kwargs)
        key[self.measure_attribute] = score
        self.insert1(key, ignore_extra_fields=True)


# @schema
# class MEIExperimentsMouse(Experiments):
#     class Restrictions(dj.Part):
#         definition = """
#         # This table contains the corresponding hashes to filter out models which form the respective experiment
#         -> master
#         -> Dataset
#         -> TrainedEnsembleModel
#         -> MEIMethod
#         -> MEI.selector_table
#         ---
#         experiment_restriction_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
#         """
#
# @schema
# class MEIExperimentsMonkey(Experiments):
#     class Restrictions(dj.Part):
#         definition = """
#         # This table contains the corresponding hashes to filter out models which form the respective experiment
#         -> master
#         -> Dataset
#         -> TrainedEnsembleModel
#         -> MEIMethod
#         -> MEIMonkey.selector_table
#         ---
#         experiment_restriction_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
#         """

@schema
class MEIExperimentsMouse(dj.Manual):
    # Table to keep track of collections of trained networks that form an experiment.
    # Instructions:
    # 1) Make an entry in Experiments with an experiment name and description
    # 2) Insert all combinations of dataset, model and trainer for this experiment name in Experiments.Restrictions.
    # 2) Populate the TrainedModel table by restricting it with Experiments.Restrictions and the experiment name.
    # 3) After training, join this table with TrainedModel and restrict by experiment name to get your results
    definition = """
    # This table contains the experiments and their descriptions
    experiment_name: varchar(100)                     # name of experiment
    ---
    -> Fabrikant.proj(experiment_fabrikant='fabrikant_name')
    experiment_comment='': varchar(2000)              # short description 
    experiment_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
    """

    class Restrictions(dj.Part):
        definition = """
        # This table contains the corresponding hashes to filter out models which form the respective experiment
        -> master
        -> Dataset
        -> TrainedEnsembleModel
        -> MEIMethod
        -> MEI.selector_table
        ---
        experiment_restriction_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
        """

    def add_entry(
            self,
            experiment_name,
            experiment_fabrikant,
            experiment_comment,
            restrictions,
            skip_duplicates=False,
    ):
        self.insert1(
            dict(
                experiment_name=experiment_name,
                experiment_fabrikant=experiment_fabrikant,
                experiment_comment=experiment_comment,
            ),
            skip_duplicates=skip_duplicates,
        )

        restrictions = [{**{"experiment_name": experiment_name}, **res} for res in restrictions]
        self.Restrictions.insert(restrictions, skip_duplicates=skip_duplicates)


# @schema
# class MEIExperimentsMonkey(dj.Manual):
#     # Table to keep track of collections of trained networks that form an experiment.
#     # Instructions:
#     # 1) Make an entry in Experiments with an experiment name and description
#     # 2) Insert all combinations of dataset, model and trainer for this experiment name in Experiments.Restrictions.
#     # 2) Populate the TrainedModel table by restricting it with Experiments.Restrictions and the experiment name.
#     # 3) After training, join this table with TrainedModel and restrict by experiment name to get your results
#     definition = """
#     # This table contains the experiments and their descriptions
#     experiment_name: varchar(100)                     # name of experiment
#     ---
#     -> Fabrikant.proj(experiment_fabrikant='fabrikant_name')
#     experiment_comment='': varchar(2000)              # short description
#     experiment_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
#     """
#
#     class Restrictions(dj.Part):
#         definition = """
#         # This table contains the corresponding hashes to filter out models which form the respective experiment
#         -> master
#         -> Dataset
#         -> TrainedEnsembleModel
#         -> MEIMethod
#         -> MEIMonkey.selector_table
#         ---
#         experiment_restriction_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
#         """
#
#     def add_entry(
#             self,
#             experiment_name,
#             experiment_fabrikant,
#             experiment_comment,
#             restrictions,
#             skip_duplicates=False,
#     ):
#         self.insert1(
#             dict(
#                 experiment_name=experiment_name,
#                 experiment_fabrikant=experiment_fabrikant,
#                 experiment_comment=experiment_comment,
#             ),
#             skip_duplicates=skip_duplicates,
#         )
#
#         restrictions = [{**{"experiment_name": experiment_name}, **res} for res in restrictions]
#         self.Restrictions.insert(restrictions, skip_duplicates=skip_duplicates)

@schema
class Gradients(dj.Computed):

    dataset_table = Dataset
    definition = """
    -> Dataset
    -> TrainedEnsembleModel
    -> MEISelector
    ---
    angle: longblob
    out_proj: longblob
    var_grad: longblob
    """

    def make(self, key, no_insert=False):
        from torch.optim import SGD
        device = "cuda"

        dataloaders, model = MEI().model_loader.load(key=key)
        output_selected_model = MEI().selector_table().get_output_selected_model(model, key).to(device)

        mean_grads, var_grads = [], []
        data_key = (MEISelector() & key).fetch1("data_key")
        d_loader = dataloaders["train"][data_key]
        for i, (images, _, _, _) in enumerate(d_loader):
            print(f"Batch number: {i+1}/{len(d_loader)}")

            images.requires_grad_()
            optimizer = SGD([images], lr=20)

            optimizer.zero_grad()
            behavior = torch.zeros((images.shape[0], 3)).to(device)
            pupil_center = torch.zeros((images.shape[0], 2)).to(device)
            x_mean = output_selected_model.predict_mean(images, behavior=behavior, pupil_center=pupil_center).sum()

            x_mean.backward()
            mean_grad = images.grad.data.cpu().numpy().copy()

            optimizer.zero_grad()
            behavior = torch.zeros((images.shape[0], 3)).to(device)
            pupil_center = torch.zeros((images.shape[0], 2)).to(device)
            x_var = output_selected_model.predict_variance(images, behavior=behavior, pupil_center=pupil_center).sum()

            x_var.backward()
            var_grad = images.grad.data.cpu().numpy().copy()
            optimizer.zero_grad()

            mean_grads.append(mean_grad)
            var_grads.append(var_grad)

        mean_grads = np.vstack(mean_grads)
        var_grads = np.vstack(var_grads)

        angle = self.angles(mean_grads.reshape(mean_grads.shape[0], -1), var_grads.reshape(var_grads.shape[0], -1))
        angle = np.degrees(angle)

        projs = self.projected(mean_grads.reshape(mean_grads.shape[0], -1), var_grads.reshape(var_grads.shape[0], -1))
        projs = projs.reshape(mean_grads.shape)

        if no_insert:
            return angle, mean_grads, var_grads, projs
        out_projs = (var_grads - projs).mean(0, keepdims=True)
        var_grads = var_grads.mean(0, keepdims=True)

        key["angle"] = angle
        key["out_proj"] = out_projs
        key["var_grad"] = var_grads
        self.insert1(key)

    def angles(self, u, v):
        assert len(u.shape) == 2, "Wrong dimensions"

        nominator = (u*v).sum(1)
        denominator = np.linalg.norm(u, axis=1)*np.linalg.norm(v, axis=1)
        return np.arccos(nominator/denominator)

    def projected(self, u, v):
        assert len(u.shape) == 2, "Wrong dimensions"

        scalar = (u*v).sum(1)
        norm = np.linalg.norm(v, axis=1)

        return v * (scalar / norm**2)[:, None]

