import filecmp
from os import path, environ
from glob import glob
from shutil import rmtree
from distutils import dir_util

from smartsim import Experiment
from smartsim.entity import NumModel
from smartsim.generation.modelwriter import ModelWriter
from smartsim.tests.decorators import modelwriter_test

@modelwriter_test
def test_write_easy_configs():

    param_dict = {
        "5": 10,                                    # MOM_input
        "FIRST": "SECOND",                          # example_input.i
        "17": 20,                                   # in.airebo
        "65": "70",                                 # in.atm
        "placeholder": "group leftupper region",    # in.crack
        "1200": "120"                               # input.nml
    }

    gen_path = "./modelwriter_test/"
    conf_path = "./test_configs/easy/marked/"
    correct_path = "./test_configs/easy/correct/"
    model = NumModel("easy", param_dict, gen_path, run_settings={})
    # copy confs to gen directory
    dir_util.copy_tree(conf_path, gen_path)
    assert(path.isdir(gen_path))

    # attach tagged files to model
    model.attach_generator_files(to_configure=glob(gen_path + "*"))

    # init modelwriter
    writer = ModelWriter()
    writer.configure_tagged_model_files(model)

    written_files = sorted(glob(gen_path + "*"))
    correct_files = sorted(glob(correct_path + "*"))

    for written, correct in zip(written_files, correct_files):
        assert(filecmp.cmp(written, correct))


@modelwriter_test
def test_write_med_configs():

    param_dict = {
        "1 0 0 0": "3 0 0 0",                         # in.ellipse.gayberne
        "'noleap'": "'leap'",                         # input.nml
        "'0 0.25 0.5 0.75 1.0'": "'1 0.25 0.5 1.0'",  # example_input.i
        '"spherical"': '"cartesian"',                 # MOM_input
        '"spoon"': '"flat"',                          # MOM_input
        "3*12.0":"3*14.0"                             # MOM_input
    }


    gen_path = "./modelwriter_test/"
    conf_path = "./test_configs/med/marked/"
    correct_path = "./test_configs/med/correct/"
    model = NumModel("med", param_dict, gen_path, run_settings={})

    # copy confs to gen directory
    dir_util.copy_tree(conf_path, gen_path)
    assert(path.isdir(gen_path))

    # attach tagged files to model
    model.attach_generator_files(to_configure=glob(gen_path + "*"))

    # init modelwriter
    writer = ModelWriter()
    writer.configure_tagged_model_files(model)

    written_files = sorted(glob(gen_path + "*"))
    correct_files = sorted(glob(correct_path + "*"))

    for written, correct in zip(written_files, correct_files):
        assert(filecmp.cmp(written, correct))


@modelwriter_test
def test_write_new_tag_configs():
    """sets the tag to the dollar sign"""

    param_dict = {
        "1 0 0 0": "3 0 0 0",                         # in.ellipse.gayberne
        "'noleap'": "'leap'",                         # input.nml
        "'0 0.25 0.5 0.75 1.0'": "'1 0.25 0.5 1.0'",  # example_input.i
        '"spherical"': '"cartesian"',                 # MOM_input
        '"spoon"': '"flat"',                          # MOM_input
        "3*12.0":"3*14.0"                             # MOM_input
    }


    gen_path = "./modelwriter_test/"
    conf_path = "./test_configs/new-tag/marked/"
    correct_path = "./test_configs/new-tag/correct/"
    model = NumModel("newtag", param_dict, gen_path, run_settings={})

    # copy confs to gen directory
    dir_util.copy_tree(conf_path, gen_path)
    assert(path.isdir(gen_path))

    # attach tagged files to model
    model.attach_generator_files(to_configure=glob(gen_path + "*"))

    # init modelwriter
    writer = ModelWriter()
    writer.set_tag("@")
    writer.configure_tagged_model_files(model)

    written_files = sorted(glob(gen_path + "*"))
    correct_files = sorted(glob(correct_path + "*"))

    for written, correct in zip(written_files, correct_files):
        assert(filecmp.cmp(written, correct))
