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
from glob import glob
import os
import pathlib
import subprocess
import sys
import typing as t
from os import path as osp

from distutils import dir_util

from smartsim._core.generation import commandgenerator

test_path = os.path.dirname(os.path.abspath(__file__))

@staticmethod
def make_test_file(
    file_name: str, file_dir: str, file_content: t.Optional[str] = None
) -> str:
    """Create a dummy file in the test output directory.

    :param file_name: name of file to create, e.g. "file.txt"
    :param file_dir: path
    :return: String path to test output file
    """
    file_path = os.path.join(file_dir, file_name)
    os.makedirs(file_dir)
    with open(file_path, "w+", encoding="utf-8") as dummy_file:
        if not file_content:
            dummy_file.write("dummy\n")
        else:
            dummy_file.write(file_content)
    return file_path


def test_symlink():
    """
    Test symlink. Make path a symlink pointing to the given path
    """
    # Prep test directory
    test_output_root = os.path.join(test_path, "tests", "test_output")
    target = pathlib.Path(test_output_root) / "sym_target_dir" / "target"
    source = pathlib.Path(test_output_root) / "sym_source"

    # Build the command
    cmd = commandgenerator.symlink_op(source, target)

    # execute the command
    subprocess.run([sys.executable, "-c", cmd])

    # Assert the two files are the same file
    assert source.is_symlink()
    assert os.readlink(source) == str(target)

    # Clean up the test directory
    os.unlink(source)


def test_copy():
    """Copy the content of the source file to the destination file."""

    test_output_root = os.path.join(test_path, "tests", "test_output")

    # make a test file with some contents
    source_file = make_test_file(
        "source.txt", pathlib.Path(test_output_root) / "source", "dummy"
    )
    dest_file = make_test_file("dest.txt", pathlib.Path(test_output_root) / "dest", "")

    # assert that source file exists, has correct contents
    assert osp.exists(source_file)
    with open(source_file, "r", encoding="utf-8") as dummy_file:
        assert dummy_file.read() == "dummy"

    cmd = commandgenerator.copy_op(source_file, dest_file)
    subprocess.run([sys.executable, "-c", cmd])

    with open(dest_file, "r", encoding="utf-8") as dummy_file:
        assert dummy_file.read() == "dummy"

    # Clean up the test directory
    os.remove(source_file)
    os.remove(dest_file)
    os.rmdir(pathlib.Path(test_output_root) / "source")
    os.rmdir(pathlib.Path(test_output_root) / "dest")


def test_move():
    """Test to move command execution"""
    test_output_root = os.path.join(test_path, "tests", "test_output")
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

    # assert that the move was successful
    assert not osp.exists(source_file)
    assert osp.exists(dest_file)
    with open(dest_file, "r", encoding="utf-8") as dummy_file:
        assert dummy_file.read() == "dummy"

    # Clean up the directories
    os.rmdir(source_dir)
    os.remove(dest_file)
    os.rmdir(dest_dir)


def test_delete():
    """Test python inline command to delete a file"""
    test_output_root = os.path.join(test_path, "tests", "test_output")

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


def test_configure(test_dir, fileutils):
    """ test configure param operations """

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
        cmd = commandgenerator.configure_op(tagged_file, param_dict)
        subprocess.run([sys.executable, "-c", cmd])

    # check that files and correct files are the same
    for written, correct in zip(tagged_files, correct_files):
        assert filecmp.cmp(written, correct)
