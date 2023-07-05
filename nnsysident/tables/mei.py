from __future__ import annotations

import os
from typing import Any, Dict

import datajoint as dj
import torch
from nnfabrik.main import Dataset, my_nnfabrik
from nnvision.tables.main import Recording
from torch.nn import Module
from torch.utils.data import DataLoader

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
