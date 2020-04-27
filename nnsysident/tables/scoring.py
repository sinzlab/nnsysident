import datajoint as dj
from nnfabrik.template import TrainedModelBase
import tempfile
import torch
import os
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from nnfabrik.utility.dj_helpers import gitlog, make_hash
import numpy as np
from .main import MonkeyExperiment
from ..utility.measures import get_oracles, get_repeats, get_FEV, get_explainable_var, get_correlations, get_poisson_loss, get_avg_correlations

schema = dj.schema(dj.config.get('schema_name', 'nnfabrik_core'))


@schema
class Correlations(dj.Computed):
    table_comment = "Trained models"

    definition = """
    -> TrainedModel
    ---
    validation_corr:            float
    test_corr:                  float
    avg_test_corr:              float
    """

    class Unit(dj.Part):
        table_comment = "Unit correlation scores"

        definition = """
        -> master
        -> MonkeyExperiment.Units
        ---
        validation_corr:            float
        test_corr:                  float
        avg_test_corr:              float
        """

    def make(self, key):
        """
        Given a key specifying the TrainedModelTable, calculates correlation scores for the model and its units.
        """

        dataloaders, model = TrainedModel().load_model(key)

        key["test_corr"]        = get_correlations(model, dataloaders=dataloaders["test"], device='cuda', per_neuron=False)
        key["validation_corr"]  = get_correlations(model, dataloaders=dataloaders["validation"], device='cuda', per_neuron=False)
        key["avg_test_corr"]    = get_avg_correlations(model, dataloaders=dataloaders["test"], device='cuda', per_neuron=False)
        self.insert1(key, ignore_extra_fields=True)


        test_corr           = get_correlations(model, dataloaders=dataloaders["test"], device='cuda', per_neuron=True, as_dict=True)
        validation_corr     = get_correlations(model, dataloaders=dataloaders["validation"], device='cuda', per_neuron=True, as_dict=True)
        avg_test_corr       = get_avg_correlations(model, dataloaders=dataloaders["test"], device='cuda', per_neuron=True, as_dict=True)

        for data_key, session_correlations in validation_corr.items():
            for i, unit_validation_correlation in enumerate(session_correlations):

                unit_id = (MonkeyExperiment.Units() & key & "session_id = '{}'".format(
                    data_key) & "unit_position = {}".format(i)).fetch1("unit_id")

                key["unit_id"] = unit_id
                key["session_id"] = data_key
                key["validation_corr"] = unit_validation_correlation
                key["test_corr"] = test_corr[data_key][i]
                key["avg_test_corr"] = avg_test_corr[data_key][i]
                self.Unit.insert1(key, ignore_extra_fields=True)


@schema
class Oracles(dj.Computed):
    table_comment = "Trained models"

    definition = """
    -> TrainedModel
    ---
    fraction_oracle:            float
    """

    def make(self, key):
        """
        Given a key specifying the TrainedModelTable, calculates Oracle scores for the model and its units.
        """

        dataloaders, model = TrainedModel().load_model(key)

        test_correlations = get_correlations(model, dataloaders=dataloaders["test"], device='cuda', per_neuron=True, as_dict=False)
        oracles = get_oracles(dataloaders=dataloaders)
        fraction_oracle, _, _, _ = np.linalg.lstsq(np.hstack(oracles)[:, np.newaxis], np.hstack(test_correlations))

        key["fraction_oracle"] = fraction_oracle
        self.insert1(key, ignore_extra_fields=True)


@schema
class StrictOracles(dj.Computed):
    table_comment = "Trained models"

    definition = """
    -> TrainedModel
    ---
    fraction_oracle:            float
    """

    class Unit(dj.Part):
        table_comment = "Unit Scores for the Strict Fraction Oracle"

        definition = """
        -> master
        -> MonkeyExperiment.Units
        ---
        fraction_oracle:        float
        """

    def make(self, key):
        """
        Given a key specifying the TrainedModelTable, calculates Oracle scores for the model and its units.
        """

        dataloaders, model = TrainedModel().load_model(key)

        test_correlations = get_correlations(model, dataloaders=dataloaders["test"], device='cuda', per_neuron=True, as_dict=False)
        oracles = get_strict_oracles(dataloaders=dataloaders)
        fraction_oracle, _, _, _ = np.linalg.lstsq(np.hstack(oracles)[:, np.newaxis], np.hstack(test_correlations))

        key["fraction_oracle"] = fraction_oracle
        self.insert1(key, ignore_extra_fields=True)
