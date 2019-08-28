from os import path

from .processor import Processor
from ..helpers import get_SSHOME
from ..error import SmartSimError, SSConfigError

import sys
import xarray as xr

class NetCDFProcessor(Processor):

    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)
        self._set_datasets()

    def get_datasets(self):
        return self.datasets

    def _set_datasets(self):
        """Use dask to open netCDF datasets
           Look for generated datasets within targets and open if present
           If not present, look through initialization arguments
        """
        try:
            targets = self._get_targets()
            if len(targets) > 0:
                data_file = self._get_config(["process","filename"])
                
                for target in targets:
                    for model in target.get_models().values():
                        data_path = path.join(model.get_path(), data_file)
                        self.datasets.append(data_path)
            else:
                # no targets, datasets must be provided
                datasets = self._get_config(["datasets"])
                # TODO catch if not a list
                for dataset in datasets:
                    ds_path = dataset
                    if not path.isfile(ds_path):
                        ds_path = path.join(get_SSHOME(), dataset)
                        if not path.isfile(ds_path):
                            raise SmartSimError(self.state.get_state(), "Could not find dataset: " + dataset)
                    self.datasets.append(ds_path)

        except SSConfigError as e:
            msg = "Datasets must either be found in target models or provided by user"
            self.log(e, level="error")
            self.log(msg, level="error")
            sys.exit()

        except SmartSimError as e:
            self.log(e, level="error")
            sys.exit()

        
                    



