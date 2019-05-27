import shutil
import glob
import sys
import subprocess
import itertools
import toml

from os import path, mkdir, getcwd
from model import Model


def read_config():
    try:
        cwd = getcwd()
        fname = cwd + "/config-sample.toml"
        with open(fname, 'r', encoding='utf-8') as fp:
            parsed_toml = toml.load(fp)
            return parsed_toml
    except Exception as e:
        print("Could not parse data generation configuration file")
        print(e)

def create_models():
    """Reads configuration for data generation in from the config.toml
       and creates models to run.

       Returns: list of high and low resolution Model objects for data
                generation
    """

    low_res_models = []
    high_res_models = []

    config = read_config()
    param_dict = config["parameters"]
    permutations = list(itertools.product(*param_dict.values()))

    params = list(param_dict.keys())
    run_settings = config["MPO_settings"]


    # Create low resolution models
    for p in permutations:
        model_params = dict(zip(params, list(p)))
        settings = config["low"]
        m = Model(model_params, settings, run_settings)
        low_res_models.append(m)

    # Create low resolution models
    for p in permutations:
        model_params = dict(zip(params, list(p)))
        settings = config["high"]
        m = Model(model_params, settings, run_settings)
        high_res_models.append(m)

    return low_res_models, high_res_models


def create_run_folders(high_res_models, low_res_models):
    raise NotImplementedError

