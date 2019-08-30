
from ..ssModule import SSModule
from ..helpers import get_SSHOME
from ..error import SmartSimError, SSConfigError


class Processor(SSModule):
    """Converts multiple files of simulation data into
       datasets convenient for machine learning experiments"""

    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)
        self.data_paths = []


###########################
### Processor Interface ###
###########################

    def get_data_paths(self):
        return self.

    def get_datasets(self):
        """Return a single dataset from each target"""
        return self.datasets

    def _set_data_paths(self):
        """Look for generated datasets within targets and open if present
           If not present, look through initialization arguments
        """
        try:
            targets = self._get_targets()
            if len(targets) > 0:
                data_file = self._get_config(["process","filename"])
                
                for target in targets:
                    for model in target.get_models().values():
                        data_path = path.join(model.get_path(), data_file)
                        self.data_paths.append(data_path)
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
                    self.data_paths.append(ds_path)

        except SSConfigError as e:
            msg = "Datasets must either be found in target models or provided by user"
            self.log(e, level="error")
            self.log(msg, level="error")
            raise SSConfigError(msg)

        except SmartSimError as e:
            self.log(e, level="error")
            raise SmartSimError(e)


###########################