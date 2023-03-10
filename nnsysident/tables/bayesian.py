import os
from nnfabrik.main import *
from nnfabrik.templates.utility import find_object
from .experiments import TrainedModel, TrainedModelTransfer, Seed

if not "stores" in dj.config:
    dj.config["stores"] = {}
dj.config["stores"]["minio_models_bayesian"] = {
    "protocol": "s3",
    "endpoint": os.environ["MINIO_ENDPOINT"],
    "bucket": "kklurzmodelsbayesian",
    "location": "dj-store",
    "access_key": os.environ["MINIO_ACCESS_KEY"],
    "secret_key": os.environ["MINIO_SECRET_KEY"],
    "secure": True,
}
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


@schema
class SeedBayesian(Seed):
    pass


@schema
class DatasetBayesian(Dataset):
    pass


@schema
class ModelBayesian(Model):
    pass


@schema
class TrainedModelBayesian(TrainedModel):
    storage = "minio_models_bayesian"
    table_comment = "Trained models for bayesian searches"

    @property
    def model_table(self):
        return ModelBayesian

    @property
    def dataset_table(self):
        return DatasetBayesian

    @property
    def seed_table(self):
        return SeedBayesian


@schema
class TrainedModelBayesianTransfer(TrainedModelTransfer):
    storage = "minio_models_bayesian"
    table_comment = "Trained models for bayesian searches"

    @property
    def model_table(self):
        return ModelBayesian

    @property
    def dataset_table(self):
        return DatasetBayesian

    @property
    def seed_table(self):
        return SeedBayesian

    def make(self, key):
        raise NotImplementedError("Still needs to be implemented")
