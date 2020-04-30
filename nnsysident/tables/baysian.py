import warnings
import datajoint as dj
import os

from nnfabrik.builder import resolve_model, get_model
from nnfabrik.utility.dj_helpers import make_hash
from nnfabrik.utility.nnf_helper import cleanup_numpy_scalar
from nnfabrik.template import TrainedModelBase
from nnfabrik.main import Fabrikant

if not 'stores' in dj.config:
    dj.config['stores'] = {}
dj.config["stores"]["minio_models_bayesian"] = {
    "protocol": "s3",
    "endpoint": os.environ["MINIO_ENDPOINT"],
    "bucket": "kklurzmodelsbayesian",
    "location": "dj-store",
    "access_key": os.environ["MINIO_ACCESS_KEY"],
    "secret_key": os.environ["MINIO_SECRET_KEY"],
}

schema = dj.schema(dj.config.get('schema_name', 'nnfabrik_core'))
print("Schema name: {}".format(dj.config["schema_name"]))

@schema
class ModelBayesian(dj.Manual):
    definition = """
    model_fn:                   varchar(64)   # name of the model function
    model_hash:                 varchar(64)   # hash of the model configuration
    ---
    model_config:               longblob      # model configuration to be passed into the function
    -> Fabrikant.proj(model_fabrikant='fabrikant_name')
    model_comment='' :          varchar(256)   # short description
    model_ts=CURRENT_TIMESTAMP: timestamp     # UTZ timestamp at time of insertion
    """

    @property
    def fn_config(self):
        model_fn, model_config = self.fetch1('model_fn', 'model_config')
        model_config = cleanup_numpy_scalar(model_config)
        return model_fn, model_config

    @staticmethod
    def resolve_fn(fn_name):
        return resolve_model(fn_name)

    def add_entry(self, model_fn, model_config, model_fabrikant=None, model_comment='', skip_duplicates=False):
        """
        Add a new entry to the model.

        Args:
            model_fn (string) - name of a callable object. If name contains multiple parts separated by `.`, this is assumed to be found in a another module and
                dynamic name resolution will be attempted. Other wise, the name will be checked inside `models` subpackage.
            model_config (dict) - Python dictionary containing keyword arguments for the model_fn
            model_fabrikant (string) - The fabrikant name. Must match an existing entry in Fabrikant table. If ignored, will attempt to resolve Fabrikant based on the database user name for the existing connection.
            model_comment - Optional comment for the entry.
            skip_duplicates - If True, no error is thrown when a duplicate entry (i.e. entry with same model_fn and model_config) is found.

        Returns:
            key - key in the table corresponding to the entry.
        """
        try:
            resolve_model(model_fn)
        except (NameError, TypeError) as e:
            warnings.warn(str(e) + '\nTable entry rejected')
            return

        if model_fabrikant is None:
            model_fabrikant = Fabrikant.get_current_user()

        model_hash = make_hash(model_config)
        key = dict(model_fn=model_fn, model_hash=model_hash, model_config=model_config,
                   model_fabrikant=model_fabrikant, model_comment=model_comment)

        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn('Corresponding entry found. Skipping...')
                key = (self & (existing)).fetch1()
            else:
                raise ValueError('Corresponding entry already exists')
        else:
            self.insert1(key)

        return key

    def build_model(self, dataloaders, seed=None, key=None):
        print('Loading model...')
        if key is None:
            key = {}
        model_fn, model_config = (self & key).fn_config

        return get_model(model_fn, model_config, dataloaders, seed=seed)

@schema
class TrainedModelBayesian(TrainedModelBase):
    table_comment = "Trained models for bayesian searches"
    model_table = ModelBayesian

    class ModelStorage(TrainedModelBase.ModelStorage):
        storage = "minio_models_bayesian"

