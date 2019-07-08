from error.ss_error import SSUnsupportedError
from writers.configwriter import ConfigWriter


class MOM6Writer(ConfigWriter):
    """Reads and writes configuration files of multiple types

       Currently Supported Types
         - namelist(nml)
         - textfile(txt)     ** only append **

       MOM6 only needs two types of configuration files to
       be written into: txt and nml
    """

    def __init__(self):
        super().__init__()
        self.config = None

    def write_config(self, param_dict, path, filetype):
        if filetype == "txt":
            self._txt(param_dict, path)
        elif filetype == "nml":
            self._nml(param_dict, path)
        else:
            raise SSUnsupportedError("Data Generation",
                                      "Configuration file type not support yet: "
                                      + filetype)

    def _nml(self, param_dict, path):
        """Edit a namelist configuration file"""

        import f90nml
        self.config = f90nml.read(path)
        for k, v in param_dict.items():
            self.deep_update(self.config, k, v)
        self.config.write(path, force=True)


    def _txt(self, param_dict, path):
        """Edit a txt based configuration file"""

        with open(path, "a+") as txt_config:
            for k, v in param_dict.items():
                txt_config.write("#override " + k + "=" + str(v) + "\n")

