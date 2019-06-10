from error.mpo_error import MpoUnsupportedError
from data_generation.writers.configwriter import ConfigWriter


class MOM6Writer(ConfigWriter):
    """Reads and writes configuration files of multiple types

       Currently Supported Types
         - namelist(nml)
         - textfile(txt)     ** only append **

      Textfile support is currently specific to MOM6
      Ideally this is the baseclass the model specific
      portions of the data-generation stage of MPO

    """

    def __init__(self):
        super().__init__()
        self.config = None

    def write_config(self, param_dict, path, filetype):
        if filetype == "txt":
            self.txt(param_dict, path)
        elif filetype == "nml":
            self.nml(param_dict, path)
        else:
            raise MpoUnsupportedError("Data Generation",
                                      "Configuration file type not support yet: "
                                      + filetype)

    def nml(self, param_dict, path):
        """Edit a namelist configuration file"""

        import f90nml
        self.config = f90nml.read(path)
        for k, v in param_dict.items():
            self.deep_update(self.config, k, v)
        self.config.write(path, force=True)


    def txt(self, param_dict, path):
        """Edit a txt based configuration file"""

        with open(path, "a+") as txt_config:
            for k, v in param_dict.items():
                txt_config.write("#override " + k + "=" + str(v) + "\n")

