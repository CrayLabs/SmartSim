from smartsim import Generator, State
from os import path, environ
from glob import glob
from shutil import rmtree
from smartsim.model import NumModel
from smartsim.generation.modelwriter import ModelWriter
from distutils import dir_util
import filecmp

def test_write_easy_configs():

    param_dict = {
        "5": 10,                                    # MOM_input
        "FIRST": "SECOND",                          # example_input.i
        "17": 20,                                   # in.airebo
        "65": "70",                                 # in.atm
        "placeholder": "group leftupper region",    # in.crack
        "1200": "120"                               # input.nml
    }

    conf_path = "./test_configs/easy/marked/"
    gen_path = "./test_configs/easy/generated/"
    correct_path = "./test_configs/easy/correct/"
    model = NumModel("easy", param_dict, gen_path, run_settings={})

    # clean up from previous test
    if path.isdir(gen_path):
        rmtree(gen_path)

    # copy confs to gen directory
    dir_util.copy_tree(conf_path, gen_path)
    assert(path.isdir(gen_path))

    # init modelwriter
    writer = ModelWriter()
    writer.write(model)

    written_files = sorted(glob(gen_path + "*"))
    correct_files = sorted(glob(correct_path + "*"))

    for written, correct in zip(written_files, correct_files):
        assert(filecmp.cmp(written, correct))

    if path.isdir(gen_path):
        rmtree(gen_path)


def test_write_med_configs():

    param_dict = {
        "1 0 0 0": "3 0 0 0",                         # in.ellipse.gayberne
        "'noleap'": "'leap'",                         # input.nml
        "'0 0.25 0.5 0.75 1.0'": "'1 0.25 0.5 1.0'",  # example_input.i
        '"spherical"': '"cartesian"',                 # MOM_input
        '"spoon"': '"flat"',                          # MOM_input
        "3*12.0":"3*14.0"                             # MOM_input
    }


    conf_path = "./test_configs/med/marked/"
    gen_path = "./test_configs/med/generated/"
    correct_path = "./test_configs/med/correct/"
    model = NumModel("med", param_dict, gen_path, run_settings={})

    # clean up from previous test
    if path.isdir(gen_path):
        rmtree(gen_path)

    # copy confs to gen directory
    dir_util.copy_tree(conf_path, gen_path)
    assert(path.isdir(gen_path))

    # init modelwriter
    writer = ModelWriter()
    writer.write(model)

    written_files = sorted(glob(gen_path + "*"))
    correct_files = sorted(glob(correct_path + "*"))

    for written, correct in zip(written_files, correct_files):
        assert(filecmp.cmp(written, correct))

    if path.isdir(gen_path):
        rmtree(gen_path)


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


    conf_path = "./test_configs/new-tag/marked/"
    gen_path = "./test_configs/new-tag/generated/"
    correct_path = "./test_configs/new-tag/correct/"
    model = NumModel("newtag", param_dict, gen_path, run_settings={})

    # clean up from previous test
    if path.isdir(gen_path):
        rmtree(gen_path)

    # copy confs to gen directory
    dir_util.copy_tree(conf_path, gen_path)
    assert(path.isdir(gen_path))

    # init modelwriter
    writer = ModelWriter()
    writer._set_tag("@")
    writer.write(model)

    written_files = sorted(glob(gen_path + "*"))
    correct_files = sorted(glob(correct_path + "*"))

    for written, correct in zip(written_files, correct_files):
        assert(filecmp.cmp(written, correct))

    if path.isdir(gen_path):
        rmtree(gen_path)
