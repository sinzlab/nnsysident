import os
import numpy as np
import datajoint as dj
from .experiments import TrainedModel, TrainedModelTransfer
from ..utility.measures import get_fraction_oracles, get_r2er, get_feve
from nnfabrik.builder import get_data
# from nnfabrik.templates.scoring import SummaryScoringBase
from nnfabrik.main import my_nnfabrik
from nnfabrik.templates.trained_model import TrainedModelBase

# create the context object
try:
    main = my_nnfabrik(os.environ["DJ_SCHEMA_NAME"])
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


class ScoringBase(dj.Computed):
    """
    Inherit from this class and decorate with your own schema to create a functional
    Score table. This serves as a template for all scores than can be computed with a
    TrainedModel and a dataloader.
    Each master table has an attribute that stores the grand average score across all Neurons.
    The `UnitScore` part table stores the score for all units.
    In order to instantiate a functional table, the default table attriubutes need to be changed.
    This template table is implementing the following logic:
    1) Loading the a trained model from the TrainedModel table. This table needs to have the 'load_model' method.
    2) gettng a dataloader. Minimally, the dataloader returns batches of inputs and targets.
        The dataloader will be built by the Datast table of nnfabrik as default.
        This table needs to have the 'get_dataloader' method
    3) passing the model and the dataloader to a scoring function, defined in the class attribute. The function is expected to
        return either the grand average score or the score per unit.
    4) The average scores and unit scores will be stored in the maser/part tables of this template.
    Attributes:
        trainedmodel_table (Datajoint Table) - an instantiation of the TrainedModelBase
        unit_table (Datajoint Table) - an instantiation of the UnitIDsBase
        scoring_function (function object) - a function that computes the average score (for the master table),
            as well as the unit scores. An example function can be found in this module under 'scoring_function_base'
        scoring_dataset (str) - following nnfabrik convention, this string specifies the key for the 'dataloaders'
            object. The dataloaders object has to contain at least ['train', 'validation', 'test'].
            This string determines, on what data tier the score is computed on. Defaults to the test set.
        scoring_attribute (str) - name of the non-primary attribute of the master and part tables for the score.
        cache (object) - Stores
    """

    nnfabrik = None

    trainedmodel_table = None
    unit_table = None
    function_kwargs = {}
    measure_dataset = "test"
    measure_attribute = "score"
    model_cache = None
    data_cache = None

    @property
    def dataset_table(self):
        return self.trainedmodel_table().dataset_table

    @staticmethod
    def measure_function(dataloaders, model, device='cuda', as_dict=True, per_neuron=True):
        raise NotImplementedError("Scoring Function has to be implemented")

    # table level comment
    table_comment = "A template table for storing results/scores of a TrainedModel"

    @property
    def definition(self):
        definition = """
                # {table_comment}
                -> self.trainedmodel_table
                ---
                {measure_attribute}:      float     # A template for a computed score of a trained model
                {measure_attribute}_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
                """.format(table_comment=self.table_comment, measure_attribute=self.measure_attribute)
        return definition

    class Units(dj.Part):
        @property
        def definition(self):
            definition = """
                # Scores for Individual Neurons
                -> master
                -> master.unit_table
                ---
                unit_{measure_attribute}:     float   # A template for a computed unit score        
                """.format(measure_attribute=self._master.measure_attribute)
            return definition

    def get_model(self, key=None):
        if self.model_cache is None:
            model = self.trainedmodel_table().load_model(key=key,
                                                         include_state_dict=True,
                                                         include_dataloader=False)
        else:
            model = self.model_cache.load(key=key,
                                          include_state_dict=True,
                                          include_dataloader=False)
        return model


    def get_dataloaders(self, key=None):
        if key is None:
            key = self.fetch1('KEY')
        dataloaders = self.dataset_table().get_dataloader(key=key) if self.data_cache is None else self.data_cache.load(key=key)
        return dataloaders[self.measure_dataset]

    def get_repeats_dataloaders(self, key=None):
        raise NotImplementedError("Function to return the repeats-dataloader has to be implemented")

    def get_avg_of_unit_dict(self, unit_scores_dict):
        return np.mean(np.hstack([v for v in unit_scores_dict.values()]))

    def insert_unit_measures(self, key, unit_measures_dict):
        key = key.copy()
        for data_key, unit_scores in unit_measures_dict.items():
            for unit_index, unit_score in enumerate(unit_scores):
                if "unit_id" in key: key.pop("unit_id")
                if "data_key" in key: key.pop("data_key")
                neuron_key = dict(unit_index=unit_index, data_key=data_key)
                unit_id = ((self.unit_table & key) & neuron_key).fetch1("unit_id")
                key["unit_id"] = unit_id
                key["unit_{}".format(self.measure_attribute)] = unit_score
                key["data_key"] = data_key
                self.Units.insert1(key, ignore_extra_fields=True)

    def make(self, key):

        dataloaders = self.get_repeats_dataloaders(key=key) if self.measure_dataset == 'test' else self.get_dataloaders(
            key=key)
        model = self.get_model(key=key)
        unit_measures_dict = self.measure_function(model=model,
                                                   dataloaders=dataloaders,
                                                   device='cuda',
                                                   as_dict=True,
                                                   per_neuron=True,
                                                   **self.function_kwargs)

        key[self.measure_attribute] = self.get_avg_of_unit_dict(unit_measures_dict)
        self.insert1(key, ignore_extra_fields=True)
        self.insert_unit_measures(key=key, unit_measures_dict=unit_measures_dict)


class SummaryScoringBase(ScoringBase):
    """
    A template scoring table with the same logic as ScoringBase, but for scores that do not have unit scores, but
    an overall score per model only.
    """
    unit_table = None
    Units = None

    def make(self, key):

        dataloaders = self.get_repeats_dataloaders(key=key) if self.measure_dataset == 'test' else self.get_dataloaders(
            key=key)
        model = self.get_model(key=key)
        key[self.measure_attribute] = self.measure_function(model=model,
                                                            dataloaders=dataloaders,
                                                            device='cuda',
                                                            **self.function_kwargs)
        self.insert1(key, ignore_extra_fields=True)



class ScoringTable(SummaryScoringBase):
    function_kwargs = {"as_dict": False, "per_neuron": False}

    def get_repeats_dataloaders(self, key=None, **kwargs):
        if key is None:
            key = self.fetch1("KEY")

        dataset_fn, dataset_config = (self.dataset_table & key).fn_config
        dataset_config["return_test_sampler"] = True
        dataset_config["tier"] = "test"
        dataset_config["seed"] = (self.trainedmodel_table().seed_table & key).fetch1("seed")

        dataloaders = get_data(dataset_fn, dataset_config)
        return dataloaders

    def make(self, key):

        dataloaders = (
            self.get_repeats_dataloaders(key=key) if self.measure_dataset == "test" else self.get_dataloaders(key=key)
        )
        model = self.get_model(key=key)
        value = self.measure_function(model=model, dataloaders=dataloaders, device="cuda", **self.function_kwargs)
        if not isinstance(value, float) and len(value) == 1:
            value = value[0]
        key[self.measure_attribute] = value
        self.insert1(key, ignore_extra_fields=True)


@schema
class OracleScore(ScoringTable):
    trainedmodel_table = TrainedModel
    measure_dataset = "test"
    measure_attribute = "fraction_oracle"
    measure_function = staticmethod(get_fraction_oracles)
    function_kwargs = {}


@schema
class OracleScoreTransfer(ScoringTable):
    trainedmodel_table = TrainedModelTransfer
    measure_dataset = "test"
    measure_attribute = "fraction_oracle"
    measure_function = staticmethod(get_fraction_oracles)
    function_kwargs = {}



@schema
class R2erScore(ScoringTable):
    trainedmodel_table = TrainedModel
    measure_dataset = "test"
    measure_attribute = "r2er"
    measure_function = staticmethod(get_r2er)
    function_kwargs = {}


@schema
class R2erScoreTransfer(ScoringTable):
    trainedmodel_table = TrainedModelTransfer
    measure_dataset = "test"
    measure_attribute = "r2er"
    measure_function = staticmethod(get_r2er)
    function_kwargs = {}


@schema
class FeveScore(ScoringTable):
    trainedmodel_table = TrainedModel
    measure_dataset = "test"
    measure_attribute = "feve"
    measure_function = staticmethod(get_feve)
    function_kwargs = {}


@schema
class FeveScoreTransfer(ScoringTable):
    trainedmodel_table = TrainedModelTransfer
    measure_dataset = "test"
    measure_attribute = "feve"
    measure_function = staticmethod(get_feve)
    function_kwargs = {}
