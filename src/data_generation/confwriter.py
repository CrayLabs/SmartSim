


class ConfWriter:
    """TODO Error handling for finding files and writing over files"""

    def __init__(self):
        self.config = None

    def write_config(self, param_dict, path, filetype):
        if filetype == "txt":
            self.txt(param_dict, path)
        elif filetype == "nml":
            self.nml(param_dict, path)
        else:
            raise Exception("File type not supported yet!")

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
                txt_config.write(k + "=" + str(v) + "\n")


    def deep_update(self, source, key, value):
        """
        Update a nested dictionary or similar mapping.
        Modify ``source`` in place.
        """
        for k, v in source.items():
            if k == key:
                source[k] = value
            elif isinstance(v, dict):
                self.deep_update(source[k], key, value)

        self.config = source

