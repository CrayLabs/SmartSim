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
import subprocess
import sys
import typing as t
from distutils import dir_util
from glob import glob
from os import path as osp

import pytest

from smartsim._core.generation import commandgenerator

test_path = os.path.dirname(os.path.abspath(__file__))
test_output_root = os.path.join(test_path, "tests", "test_output")


def test_symlink_entity():
    """
    Test the execution of the python script created by symlink_op
    """

    source = pathlib.Path(test_output_root) / "sym_source"
    os.mkdir(source)

    # entity_path to be the dest dir
    entity_path = os.path.join(test_output_root, "entity_name")
    os.mkdir(entity_path)

    target = pathlib.Path(test_output_root) / "entity_name" / "sym_source"

    # # Build the command
    cmd = commandgenerator.symlink_op(source, entity_path)
    # # execute the command
    subprocess.run([sys.executable, "-c", cmd])

    # # Assert the two files are the same file
    assert target.is_symlink()
    assert os.readlink(target) == str(source)

    # Clean up the test directory
    os.unlink(target)
    os.rmdir(pathlib.Path(test_output_root) / "entity_name")
    os.rmdir(pathlib.Path(test_output_root) / "sym_source")


def test_copy_op_file():
    """Test the execution of the python script created by copy_op
    Test the operation to copy the content of the source file to the destination path
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
    cmd = commandgenerator.copy_op(source_file, entity_path)
    subprocess.run([sys.executable, "-c", cmd])

    # clean up
    os.remove(pathlib.Path(to_copy) / "copy_file.txt")
    os.rmdir(pathlib.Path(test_output_root) / "to_copy")

    os.remove(pathlib.Path(entity_path) / "copy_file.txt")
    os.rmdir(pathlib.Path(test_output_root) / "entity_name")


def test_copy_op_dirs():
    """Test the execution of the python script created by copy_op
    Test the oeprations that copies an entire directory tree source to a new location destination
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

    cmd = commandgenerator.copy_op(to_copy, entity_path)
    subprocess.run([sys.executable, "-c", cmd])

    # Clean up
    os.remove(pathlib.Path(to_copy) / "copy_file.txt")
    os.remove(pathlib.Path(to_copy) / "copy_file_2.txt")
    os.rmdir(pathlib.Path(test_output_root) / "to_copy")
    os.remove(pathlib.Path(entity_path) / "copy_file.txt")
    os.remove(pathlib.Path(entity_path) / "copy_file_2.txt")
    os.rmdir(pathlib.Path(test_output_root) / "entity_name")


def test_copy_op(fileutils):
    """Test the execution of the python script created by copy_op
    Test the operation to copy the content of the source file to the destination file.
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

    cmd = commandgenerator.copy_op(to_copy_file, entity_path)
    subprocess.run([sys.executable, "-c", cmd])

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
    """Test the execution of the python script created by copy_op
    Test that a FileNotFoundError is raised when there is a bad source file
    """

    to_copy = os.path.join(test_output_root, "to_copy")
    os.mkdir(to_copy)
    entity_path = os.path.join(test_output_root, "entity_name")
    os.mkdir(entity_path)

    # Execute copy
    with pytest.raises(FileNotFoundError) as ex:
        commandgenerator.copy_op("/not/a/real/path", entity_path)
    assert "is not a valid path" in ex.value.args[0]
    # clean up
    os.rmdir(pathlib.Path(test_output_root) / "to_copy")
    os.rmdir(pathlib.Path(test_output_root) / "entity_name")


def test_copy_op_bad_dest_path():
    """Test the execution of the python script created by copy_op.
    Test that a FileNotFoundError is raised when there is a bad destination file."""

    to_copy = os.path.join(test_output_root, "to_copy")
    os.mkdir(to_copy)

    source_file = pathlib.Path(to_copy) / "copy_file.txt"
    entity_path = os.path.join(test_output_root, "entity_name")
    os.mkdir(entity_path)

    with pytest.raises(FileNotFoundError) as ex:
        commandgenerator.copy_op(source_file, "/not/a/real/path")
    assert "is not a valid path" in ex.value.args[0]

    # clean up
    os.rmdir(pathlib.Path(test_output_root) / "to_copy")
    os.rmdir(pathlib.Path(test_output_root) / "entity_name")


def test_move_op():
    """Test the execution of the python script created by move_op.
    Test the operation to move a file"""

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

    cmd = commandgenerator.move_op(source_file, dest_file)
    subprocess.run([sys.executable, "-c", cmd])

    # Assert that the move was successful
    assert not osp.exists(source_file)
    assert osp.exists(dest_file)
    with open(dest_file, "r", encoding="utf-8") as dummy_file:
        assert dummy_file.read() == "dummy"

    # Clean up the directories
    os.rmdir(source_dir)
    os.remove(dest_file)
    os.rmdir(dest_dir)


def test_delete_op():
    """Test the execution of the python script created by delete_op.
    Test the operation to delete a file"""

    # Make a test file with dummy text
    to_del = pathlib.Path(test_output_root) / "app_del.txt"
    with open(to_del, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("dummy")

    assert osp.exists(to_del)
    with open(to_del, "r", encoding="utf-8") as dummy_file:
        assert dummy_file.read() == "dummy"

    cmd = commandgenerator.delete_op(to_del)
    subprocess.run([sys.executable, "-c", cmd])

    # Assert file has been deleted
    assert not osp.exists(to_del)


def test_delete_op_bad_path():
    """Test the execution of the python script created by delete_op.
    Test that FileNotFoundError is raised when a bad path is given to the
    soperation to delete a file"""

    test_output_root = os.path.join(test_path, "tests", "test_output")
    to_del = pathlib.Path(test_output_root) / "not_real.txt"

    with pytest.raises(FileNotFoundError) as ex:
        commandgenerator.delete_op(to_del)
    assert "is not a valid path" in ex.value.args[0]


def test_configure_op(test_dir, fileutils):
    """Test the execution of the python script created by configure_op.
    Test configure param operations with a tag parameter given"""

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
        cmd = commandgenerator.configure_op(tagged_file, param_dict, tag)
        subprocess.run([sys.executable, "-c", cmd])

    # check that files and correct files are the same
    for written, correct in zip(tagged_files, correct_files):
        assert filecmp.cmp(written, correct)


def test_configure_op_no_tag(test_dir, fileutils):
    """Test the execution of the python script created by configure_op.
    Test configure param operations with no tag parameter given"""

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
        cmd = commandgenerator.configure_op(tagged_file, param_dict, False)
        subprocess.run([sys.executable, "-c", cmd])

    # check that files and correct files are the same
    for written, correct in zip(tagged_files, correct_files):
        assert filecmp.cmp(written, correct)
