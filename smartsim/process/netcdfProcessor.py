from os import path
import xarray as xr

from .processor import Processor

class NetCDFProcessor(Processor):

    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)
        self._set_datasets()



        
                    



