# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import filecmp
import os
import pathlib

import typing as t
from distutils import dir_util
from glob import glob
from os import path as osp

import pytest

from smartsim._core.generation import commandgenerator
from smartsim._core.entrypoints import command_generator

test_path = os.path.dirname(os.path.abspath(__file__))
test_output_root = os.path.join(test_path, "tests", "test_output")


def test_symlink_files():
    """
    Test operation to symlink files
    """

    source = pathlib.Path(test_output_root) / "sym_source"
    os.mkdir(source)
    source_file = pathlib.Path(source) / "sym_source.txt"
    with open(source_file, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("")
    # entity_path to be the dest dir
    entity_path = os.path.join(test_output_root, "entity_name")

    command_generator.symlink_op(source_file, entity_path)

    link = pathlib.Path(test_output_root) / "entity_name"
    # Assert the two files are the same file
    assert link.is_symlink()
    assert os.readlink(link) == str(source_file)
    
    # Clean up the test directory
    os.unlink(link)
    os.remove(pathlib.Path(source) / "sym_source.txt")
    os.rmdir(pathlib.Path(test_output_root) / "sym_source") 


def test_symlink_dir():
    """
    Test operation to symlink directories
    """

    source = pathlib.Path(test_output_root) / "sym_source"
    os.mkdir(source)

    # entity_path to be the dest dir
    entity_path = os.path.join(test_output_root, "entity_name")

    command_generator.symlink_op(source, entity_path)

    link = pathlib.Path(test_output_root) / "entity_name"
    # Assert the two files are the same file
    assert link.is_symlink()
    assert os.readlink(link) == str(source)

    # Clean up the test directory
    os.unlink(link)
    os.rmdir(pathlib.Path(test_output_root) / "sym_source") 


def test_copy_op_file():
    """Test the operation to copy the content of the source file to the destination path
    with an empty file of the same name already in the directory"""

    to_copy = os.path.join(test_output_root, "to_copy")
    os.mkdir(to_copy)

    source_file = pathlib.Path(to_copy) / "copy_file.txt"
    with open(source_file, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("dummy")

    entity_path = os.path.join(test_output_root, "entity_name")
    os.mkdir(entity_path)

    dest_file = os.path.join(test_output_root, "entity_name", "copy_file.txt")
    with open(dest_file, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("")

    # Execute copy
    command_generator.copy_op(source_file, dest_file)

    # clean up
    os.remove(pathlib.Path(to_copy) / "copy_file.txt")
    os.rmdir(pathlib.Path(test_output_root) / "to_copy")

    os.remove(pathlib.Path(entity_path) / "copy_file.txt")
    os.rmdir(pathlib.Path(test_output_root) / "entity_name")


def test_copy_op_dirs():
    """Test the oeprations that copies an entire directory tree source to a new location destination
    """

    to_copy = os.path.join(test_output_root, "to_copy")
    os.mkdir(to_copy)

    # write some test files in the dir
    source_file = pathlib.Path(to_copy) / "copy_file.txt"
    with open(source_file, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("dummy1")

    source_file_2 = pathlib.Path(to_copy) / "copy_file_2.txt"
    with open(source_file_2, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("dummy2")

    # entity_path to be the dest dir
    entity_path = os.path.join(test_output_root, "entity_name")
    os.mkdir(entity_path)
    # copy those files?

    cmd = command_generator.copy_op(to_copy, entity_path)
    #subprocess.run([sys.executable, "-c", cmd])

    # Clean up
    os.remove(pathlib.Path(to_copy) / "copy_file.txt")
    os.remove(pathlib.Path(to_copy) / "copy_file_2.txt")
    os.rmdir(pathlib.Path(test_output_root) / "to_copy")
    os.remove(pathlib.Path(entity_path) / "copy_file.txt")
    os.remove(pathlib.Path(entity_path) / "copy_file_2.txt")
    os.rmdir(pathlib.Path(test_output_root) / "entity_name")


def test_copy_op(fileutils):
    """Test the operation to copy the content of the source file to the destination file.
    """

    # make a test file with some contents
    to_copy_file = fileutils.make_test_file(
        "to_copy.txt", pathlib.Path(test_output_root) / "to_copy", "dummy"
    )

    entity_path = os.path.join(test_output_root, "entity_name")

    os.mkdir(entity_path)

    # assert that source file exists, has correct contents
    assert osp.exists(to_copy_file)
    with open(to_copy_file, "r", encoding="utf-8") as dummy_file:
        assert dummy_file.read() == "dummy"

    cmd = command_generator.copy_op(to_copy_file, entity_path)
    #subprocess.run([sys.executable, "-c", cmd])

    entity_file = os.path.join(test_output_root, "entity_name", "to_copy.txt")
    # asser that the entity_path now has the source file with the correct contents
    with open(entity_file, "r", encoding="utf-8") as dummy_file:
        assert "dummy" in dummy_file.read()

    # Clean up the test directory
    os.remove(to_copy_file)
    os.remove(entity_file)
    os.rmdir(pathlib.Path(test_output_root) / "to_copy")
    os.rmdir(pathlib.Path(test_output_root) / "entity_name")


def test_copy_op_bad_source_file():
    """Test that a FileNotFoundError is raised when there is a bad source file
    """

    to_copy = os.path.join(test_output_root, "to_copy")
    os.mkdir(to_copy)
    entity_path = os.path.join(test_output_root, "entity_name")
    os.mkdir(entity_path)

    # Execute copy
    with pytest.raises(FileNotFoundError) as ex:
        command_generator.copy_op("/not/a/real/path", entity_path)
    assert "is not a valid path" in ex.value.args[0]
    # clean up
    os.rmdir(pathlib.Path(test_output_root) / "to_copy")
    os.rmdir(pathlib.Path(test_output_root) / "entity_name")


def test_copy_op_bad_dest_path():
    """Test that a FileNotFoundError is raised when there is a bad destination file.
    """

    to_copy = os.path.join(test_output_root, "to_copy")
    os.mkdir(to_copy)

    source_file = pathlib.Path(to_copy) / "copy_file.txt"
    entity_path = os.path.join(test_output_root, "entity_name")
    os.mkdir(entity_path)

    with pytest.raises(FileNotFoundError) as ex:
        command_generator.copy_op(source_file, "/not/a/real/path")
    assert "is not a valid path" in ex.value.args[0]

    # clean up
    os.rmdir(pathlib.Path(test_output_root) / "to_copy")
    os.rmdir(pathlib.Path(test_output_root) / "entity_name")


def test_move_op():
    """Test the operation to move a file"""

    source_dir = os.path.join(test_output_root, "from_here")
    os.mkdir(source_dir)
    dest_dir = os.path.join(test_output_root, "to_here")
    os.mkdir(dest_dir)

    dest_file = os.path.join(test_output_root, "to_here", "to_here.txt")
    source_file = pathlib.Path(source_dir) / "app_move.txt"

    with open(source_file, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("dummy")

    assert osp.exists(source_file)
    with open(source_file, "r", encoding="utf-8") as dummy_file:
        assert dummy_file.read() == "dummy"

    cmd = command_generator.move_op(source_file, dest_file)
    #subprocess.run([sys.executable, "-c", cmd])

    # Assert that the move was successful
    assert not osp.exists(source_file)
    assert osp.exists(dest_file)
    with open(dest_file, "r", encoding="utf-8") as dummy_file:
        assert dummy_file.read() == "dummy"

    # Clean up the directories
    os.rmdir(source_dir)
    os.remove(dest_file)
    os.rmdir(dest_dir)




def test_configure_op(test_dir, fileutils):
    """Test configure param operations with a tag parameter given"""

    # the param dict for configure operations
    param_dict = {
        "5": 10,  # MOM_input
        "FIRST": "SECOND",  # example_input.i
        "17": 20,  # in.airebo
        "65": "70",  # in.atm
        "placeholder": "group leftupper region",  # in.crack
        "1200": "120",  # input.nml
    }
    tag = ";"
    # retreive tagged files
    conf_path = fileutils.get_test_conf_path(osp.join("tagged_tests", "marked/"))
    # retrieve files to compare after test
    correct_path = fileutils.get_test_conf_path(osp.join("tagged_tests", "correct/"))

    # copy files to test directory
    dir_util.copy_tree(conf_path, test_dir)
    assert osp.isdir(test_dir)

    tagged_files = sorted(glob(test_dir + "/*"))
    correct_files = sorted(glob(correct_path + "*"))

    # Run configure op on test files
    for tagged_file in tagged_files:
        cmd = command_generator.configure_op(tagged_file,dest=tagged_file, params=param_dict,tag_delimiter=tag)
        #subprocess.run([sys.executable, "-c", cmd])

    # check that files and correct files are the same
    for written, correct in zip(tagged_files, correct_files):
        assert filecmp.cmp(written, correct)


def test_configure_op_no_tag(test_dir, fileutils):
    """Test configure param operations with no tag parameter given"""

    # the param dict for configure operations
    param_dict = {
        "5": 10,  # MOM_input
        "FIRST": "SECOND",  # example_input.i
        "17": 20,  # in.airebo
        "65": "70",  # in.atm
        "placeholder": "group leftupper region",  # in.crack
        "1200": "120",  # input.nml
    }
    # retreive tagged files
    conf_path = fileutils.get_test_conf_path(osp.join("tagged_tests", "marked/"))
    # retrieve files to compare after test
    correct_path = fileutils.get_test_conf_path(osp.join("tagged_tests", "correct/"))

    # copy files to test directory
    dir_util.copy_tree(conf_path, test_dir)
    assert osp.isdir(test_dir)

    tagged_files = sorted(glob(test_dir + "/*"))
    correct_files = sorted(glob(correct_path + "*"))

    # Run configure op on test files
    for tagged_file in tagged_files:
        cmd = command_generator.configure_op(tagged_file,dest=None, params=param_dict)
       # subprocess.run([sys.executable, "-c", cmd])

    # check that files and correct files are the same
    for written, correct in zip(tagged_files, correct_files):
        assert filecmp.cmp(written, correct)

def test_delete_op():
    """Test the operation to delete a file"""

    # Make a test file with dummy text
    to_del = pathlib.Path(test_output_root) / "app_del.txt"
    with open(to_del, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("dummy")

    assert osp.exists(to_del)
    with open(to_del, "r", encoding="utf-8") as dummy_file:
        assert dummy_file.read() == "dummy"

    cmd = command_generator.delete_op(to_del)
    #subprocess.run([sys.executable, "-c", cmd])

    # Assert file has been deleted
    assert not osp.exists(to_del)


def test_delete_op_bad_path():
    """Test that FileNotFoundError is raised when a bad path is given to the
    soperation to delete a file"""

    test_output_root = os.path.join(test_path, "tests", "test_output")
    to_del = pathlib.Path(test_output_root) / "not_real.txt"

    with pytest.raises(FileNotFoundError) as ex:
        command_generator.delete_op(to_del)
    assert "is not a valid path" in ex.value.args[0]


from smartsim._core.entrypoints.telemetrymonitor import get_parser
from smartsim._core.entrypoints.command_generator import get_parser 

def test_parser_remove():
    """Test that the parser succeeds when receiving expected args"""
    
    
    parser = get_parser()

    # Make a test file with dummy text
    to_remove = pathlib.Path(test_output_root) / "app_del.txt"
    with open(to_remove, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("dummy")

    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                 help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                 const=sum, default=max,
    #                 help='sum the integers (default: find the max)')

    #args = parser.parse_args()



   # test_dir = "/foo/bar"
   # test_freq = 123

   

   # cmd = f"-exp_dir {test_dir} -frequency {test_freq}"
    cmd = f"_core/entrypoints/file_operations.py remove {to_remove}"
    args = cmd.split()

    print(args)

    #ns = parser.parse_args(args)

   # assert ns.exp_dir == test_dir
   # assert ns.frequency == test_freq

   #assert not osp.exists(to_del)



ALL_ARGS = {"-exp_dir", "-frequency"}

@pytest.mark.parametrize(
    ["cmd", "missing"],
    [
        pytest.param("", {"-exp_dir", "-frequency"}, id="no args"),
        pytest.param("-exp_dir /foo/bar", {"-frequency"}, id="no freq"),
        pytest.param("-frequency 123", {"-exp_dir"}, id="no dir"),
    ],
)

def test_parser_reqd_args(capsys, cmd, missing):
    """Test that the parser reports any missing required arguments"""
    parser = get_parser()

    args = cmd.split()

    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as ex:
        ns = parser.parse_args(args)

    captured = capsys.readouterr()
    assert "the following arguments are required" in captured.err
    err_desc = captured.err.split("the following arguments are required:")[-1]
    for arg in missing:
        assert arg in err_desc

    expected = ALL_ARGS - missing
    for exp in expected:
        assert exp not in err_desc


def test_parser():
    """Test that the parser succeeds when receiving expected args"""
    parser = get_parser()

    test_dir = "/foo/bar"
    test_freq = 123

    cmd = f"-exp_dir {test_dir} -frequency {test_freq}"
    args = cmd.split()

    ns = parser.parse_args(args)
    print(ns)

    assert ns.exp_dir == test_dir
    assert ns.frequency == test_freq





def test_parser_move():
    """Test that the parser succeeds when receiving expected args"""
    parser = get_parser()


    src_path = "/absolute/file/src/path"
    dest_path = "/absolute/file/dest/path"

    cmd = f"move {src_path} {dest_path}"    # must be absolute path
    # python 
    args = cmd.split()
    print(args)
    ns = parser.parse_args(args)
    print(ns)

    assert ns.src_path == src_path
    assert ns.dest_path == dest_path

def test_parser_remove():
    """Test that the parser succeeds when receiving expected args"""
    parser = get_parser()

    file_path = "/absolute/file/path"
    cmd = f"remove {file_path}"

    args = cmd.split()
    ns = parser.parse_args(args)
    assert ns.to_remove == file_path

def test_parser_symlink():
    """Test that the parser succeeds when receiving expected args"""
    parser = get_parser()

    src_path = "/absolute/file/src/path"
    dest_path = "/absolute/file/dest/path"
    cmd = f"symlink {src_path} {dest_path}"

    args = cmd.split()

    ns = parser.parse_args(args)

    assert ns.source_path == src_path
    assert ns.dest_path == dest_path

def test_parser_copy():
    """Test that the parser succeeds when receiving expected args"""
    parser = get_parser()

    src_path = "/absolute/file/src/path"
    dest_path = "/absolute/file/dest/path"

    cmd = f"copy {src_path} {dest_path}"    # must be absolute path
    # python 
    args = cmd.split()

    ns = parser.parse_args(args)
    print(ns)

    assert ns.source_path == src_path
    assert ns.dest_path == dest_path

import json

def test_parser_configure():
    """Test that the parser succeeds when receiving expected args"""
    parser = get_parser()

    src_path = "/absolute/file/src/path"
    dest_path = "/absolute/file/dest/path"
    tag_delimiter = ";"
    params = '{"5":10}'
    
#     '{"5": 10}'

#     >>>
# {u'value1': u'key1'}

    cmd = f"configure {src_path} {dest_path} {tag_delimiter} {params}"    # must be absolute path
    # python 
    args = cmd.split()

    ns = parser.parse_args(args)
    print(ns)


#parser.add_argument('-m', '--my-dict', type=str)
#args = parser.parse_args()
    print("HIHIHI")
#import json
    my_dictionary = json.loads(ns.param_dict)
    print(my_dictionary)
    print("is thre a type",type(my_dictionary))


    assert ns.source_path == src_path
    assert ns.dest_path == dest_path
    assert ns.tag_delimiter == tag_delimiter
    assert ns.param_dict == params


def test_command_generator_entrypoint():
    ...
    parser = get_parser()
 
   # from smartsim._core.generation import commandgenerator
