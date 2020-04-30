import os
import datajoint as dj

from nnfabrik.template import TrainedModelBase


if not 'stores' in dj.config:
    dj.config['stores'] = {}
dj.config["stores"]["minio_models"] = {
    "protocol": "s3",
    "endpoint": os.environ["MINIO_ENDPOINT"],
    "bucket": "kklurzmodels",
    "location": "dj-store",
    "access_key": os.environ["MINIO_ACCESS_KEY"],
    "secret_key": os.environ["MINIO_SECRET_KEY"],
}

schema = dj.schema(dj.config.get('schema_name', 'nnfabrik_core'))

@schema
class TrainedModel(TrainedModelBase):
    table_comment = "Trained models"

    class ModelStorage(TrainedModelBase.ModelStorage):
        storage = "minio_models"


# @schema
# class Experiments(dj.Manual):
#     definition = """
#     """
