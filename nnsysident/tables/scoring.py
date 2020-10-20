import datajoint as dj
from .experiments import TrainedModel, TrainedModelTransfer
from ..utility.measures import get_fraction_oracles

from nnfabrik.utility.dj_helpers import CustomSchema
from nnfabrik.builder import get_data
from nnfabrik.template import SummaryScoringBase

schema = CustomSchema(dj.config.get("schema_name", "nnfabrik_core"))


class ScoringTable(SummaryScoringBase):
    function_kwargs = {"as_dict": False, "per_neuron": False}

    def get_repeats_dataloaders(self, key=None, **kwargs):
        if key is None:
            key = self.fetch1("KEY")

        dataset_fn, dataset_config = (self.dataset_table & key).fn_config
        dataset_config["return_test_sampler"] = True
        dataset_config["tier"] = "test"
        dataset_config["seed"] = (self.trainedmodel_table.seed_table & key).fetch1("seed")

        dataloaders = get_data(dataset_fn, dataset_config)
        return dataloaders

    def make(self, key):

        dataloaders = (
            self.get_repeats_dataloaders(key=key) if self.measure_dataset == "test" else self.get_dataloaders(key=key)
        )
        model = self.get_model(key=key)[1]
        value = self.measure_function(model=model, dataloaders=dataloaders, device="cuda", **self.function_kwargs)
        if type(value) != float and len(value) == 1:
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
