import abc

class ConfigWriter(abc.ABC):
    """Reads and writes configuration files of multiple types

       Currently Supported Types
         - namelist(nml)
         - textfile(txt)     ** only append **

      Textfile support is currently specific to MOM6
      Ideally this is the baseclass the model specific
      portions of the data-generation stage of MPO

    """


    def __init__(self):
        """ init a new config writer"""
        self.config = None

    @abc.abstractmethod
    def write_config(self, param_dict, path, filetype):
        """The only method that must be written over to write to
           a specific type of configuration file

           Args
               param_dict  (dict): a key value store of the parameter
                                    and the name of that parameter
               path        (str) : The path to the configuration file
               filetype    (str) : The type of the config file"""
        pass


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
