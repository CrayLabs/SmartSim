from os import listdir, getcwd, mkdir, path
from sklearn import preprocessing
from functools import partial


import sys
import xarray as xr
import numpy as np
import pandas as pd
import multiprocessing

from smartsim import Processor
from smartsim.error import SmartSimError


class mom6Processor(Processor):

    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)
        self.multilayer_features = ["KE"] #["stream", "KE"]
        self.one_layer_features = [] #["SSH", "speed"]
        self._set_arguments()

    def _set_arguments(self):
        self.total_years = self.get_config(["years_simmed"])
        self.years_per_sample = self.get_config(["years_per_sample"])
        multi_layer = self.get_config("multilayer_features", none_ok=True)
        one_layer = self.get_config("one_layer_features", none_ok=True)
        if multi_layer:
            self.multilayer_features = multi_layer
        if one_layer:
            self.one_layer_features = one_layer

    def process(self, num_proc=4):
        run_paths = self.get_data(filename="ocean_mean_annual.nc")
        self.sample_count = len(run_paths) * (self.total_years/self.years_per_sample)
        pool = multiprocessing.Pool(num_proc)
        datasets = pool.map(partial(self.preprocess_data), run_paths)
        self.write_dataset(datasets)


    def preprocess_data(self, run_info):
        """Conduct preprocessing of a single model simulation. The preprocessing
           is written in parallel using multi-processing.

        Args
            run_info (dict{str, str}): path to directory that contains simulation output

        """
        model_name, path = run_info
        self.log("Processing " + model_name)

        # get parameter values
        model = self.get_model(model_name)
        kh = model.get_param_value("KH")
        khth = model.get_param_value("KHTH")
        param_dict = {"KH": int(kh), "KHTH": int(khth)}

        run_data = self.collect_run_data(path)
        filled_state_tensors = self.create_training_samples(run_data)
        averaged_state_tensors = self.average_by_year(filled_state_tensors)
        normalized_state_tensors = self.normalize_data(averaged_state_tensors)
        with_targets = self.add_targets(normalized_state_tensors, **param_dict)
        return with_targets

    def add_targets(self, tensors, **kwargs):
        """Adds targets to each sample"""

        with_targets = []
        for tensor in tensors:
            for name, value in kwargs.items():
                tensor[name] = value
            with_targets.append(tensor)
        return with_targets


    def collect_run_data(self, parent_path):
        """Collect the NetCDF dataset for a single model run.
           Splits the years of model simulation into single years.

           Args
            parent_path (str): parent directory filepath of model configs

           Returns
            list: a list of single years of a model configuration
        """
        run_data = []
        data = xr.open_dataset(parent_path, decode_times=False)
        data_by_year = list(data.groupby("time"))
        for year in range(self.total_years):
            run_data.append(data_by_year[year][1])
        return run_data


    def shape_training_data(self, run_data):
        """Create state tensors to hold output of the MOM6 model
           state tensor = (lat + long) x len(state_vars)

           Args
            run_data (str): all single year samples of simulation data

           Returns
            A list of empty state tensors to be filled with MOM6 model data
            Also returns information about the model run itself.
        """
        data = run_data[0]

        # get number of columns per grid point
        num_state_vars = len(self.multilayer_features)
        num_one_layer = len(self.one_layer_features)

        # Dimensions of the data
        layers = len(data.zl)
        latitude = data.yh.size
        longitude = data.xh.size
        grid_points = latitude * longitude
        model_run_info = (layers, latitude, longitude)

        # Create arrays which will become the state tensor
        state_tensors = []
        for _ in range(len(run_data)):
            state_tensors.append(np.zeros((grid_points, (num_state_vars * layers) + num_one_layer)))


        return state_tensors, model_run_info


    def create_training_samples(self, run_data):
        """Create and fill tensors to be used as samples in training.
            Extracts features from the MOM6 model netCDF output.
            Calculate streamfunction if within multi_layer_features.

           Args
            run_data (str): parent directory filepath of MOM6 models

           Returns
            list of filled state tensors that are created by shape_training_data

        """

        # get state tensors for each run to hold training data
        state_tensor_list, data_info = self.shape_training_data(run_data)

        layers = data_info[0]
        grid_points = data_info[1] * data_info[2]

        # Calculate zonal streamfunction
        if "stream" in self.multilayer_features:
            for i in range(len(run_data)):
                data = run_data[i]
                run_data[i]["stream"] = data.vh.cumsum('xh')

        header = []
        for i, data in enumerate(run_data):
            # Loop over all state variables to fill state tensor

            state_tensor = state_tensor_list[i]
            ncol = 0
            for layer in range(0, layers):
                for var in self.multilayer_features:
                    feature_name = var + "_l" + str(layer)
                    if feature_name not in header:
                        header.append(feature_name)
                    state_tensor[: , ncol] = np.array(data[var][layer, :]).reshape(grid_points)
                    ncol += 1
            for var in self.one_layer_features:
                feature_name = var
                if feature_name not in header:
                    header.append(feature_name)
                state_tensor[: , ncol] = np.array(data[var]).reshape(grid_points)
                ncol += 1

            state_tensor_list[i] = pd.DataFrame(state_tensor, columns=header)
        
        return state_tensor_list


    def average_by_year(self, state_tensors):
        """Takes in list of state_tensors to be averaged by num years of simulation

           Args
            state_tensors (list): list of feature filled dataframes to be averged

           Returns
            list of averaged state tensors averaged for num_years number of years

        """
        averaged_state_tensors = [pd.concat((state_tensors[x:x+self.years_per_sample])).groupby(level=0).mean()
                                    for x in range(self.total_years) if x % (self.years_per_sample) == 0]

        return averaged_state_tensors



    def normalize_data(self, state_tensors):
        """Create normal distribution of data in each time averaged sample

           Args
            state_tensors (list): list of training samples to be normalized.

           Returns
            normalized training samples
        """

        normalized_state_tensors = []
        for state_tensor in state_tensors:
            norm_tensor = ((state_tensor-state_tensor.min())/(state_tensor.max()-state_tensor.min()))
            normalized_state_tensors.append(norm_tensor)
        return normalized_state_tensors


    def write_dataset(self, datasets):
        """Format final dataset shape by reshaping normalized state tensors into flat
           np arrays. append to final dataset and add targets. Also creates header and
           saves to a csv.

           Args    
            datasets (array): an array of arrays that hold pandas dataframes for each sample.
        """
        samples = []
        feature_names = datasets[0][0].columns
        feature_names = feature_names.drop("KH")
        feature_names = feature_names.drop("KHTH")

        # remove targets, reshape, add targets back in.
        for dataset in datasets:
            for ds in dataset:

                kh = ds["KH"][0]
                khth = ds["KHTH"][0]
                ds = ds.drop(["KH", "KHTH"], axis=1)
                n = np.array(ds)
                np_tensor = np.array(ds)
                np_data = np_tensor.reshape(1, -1)
                np_data = np.append(np_data, kh)
                np_data = np.append(np_data, khth)
                samples.append(np_data)


        # set feature names
        data = np.array(samples).reshape(int(self.sample_count), -1)
        df = pd.DataFrame(data)
        features = ""
        grid_point = 0

        # Create header for a more informative dataset
        for i in range(1, data.shape[-1]-2):
            if i % len(feature_names) == 0:
                grid_point += 1
            features += feature_names[i%len(feature_names)] + "_gp" + str(grid_point) + ","
        features += "KH,"
        features += "KHTH"

        exp_path = self.get_experiment_path()
        f_name = path.join(exp_path, "dataset.csv")
        np.savetxt(f_name, data, delimiter=",", comments='', header=features)
