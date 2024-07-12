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

import base64
import filecmp
import os
import pathlib
import pickle
from distutils import dir_util
from glob import glob
from os import path as osp

import pytest

from smartsim._core.entrypoints import file_operations
from smartsim._core.entrypoints.file_operations import get_parser


def test_symlink_files(test_dir):
    """
    Test operation to symlink files
    """
    # Set source directory and file
    source = pathlib.Path(test_dir) / "sym_source"
    os.mkdir(source)
    source_file = pathlib.Path(source) / "sym_source.txt"
    with open(source_file, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("")

    # Set path to be the destination directory
    entity_path = os.path.join(test_dir, "entity_name")

    parser = get_parser()
    cmd = f"symlink {source_file} {entity_path}"
    args = cmd.split()
    ns = parser.parse_args(args)

    file_operations.symlink(ns)

    # Assert the two files are the same file
    link = pathlib.Path(test_dir) / "entity_name"
    assert link.is_symlink()
    assert os.readlink(link) == str(source_file)

    # Clean up the test directory
    os.unlink(link)
    os.remove(pathlib.Path(source) / "sym_source.txt")
    os.rmdir(pathlib.Path(test_dir) / "sym_source")


def test_symlink_dir(test_dir):
    """
    Test operation to symlink directories
    """

    source = pathlib.Path(test_dir) / "sym_source"
    os.mkdir(source)

    # entity_path to be the dest dir
    entity_path = os.path.join(test_dir, "entity_name")

    parser = get_parser()
    cmd = f"symlink {source} {entity_path}"
    args = cmd.split()
    ns = parser.parse_args(args)

    file_operations.symlink(ns)

    link = pathlib.Path(test_dir) / "entity_name"
    # Assert the two files are the same file
    assert link.is_symlink()
    assert os.readlink(link) == str(source)

    # Clean up the test directory
    os.unlink(link)
    os.rmdir(pathlib.Path(test_dir) / "sym_source")


def test_copy_op_file(test_dir):
    """Test the operation to copy the content of the source file to the destination path
    with an empty file of the same name already in the directory"""

    to_copy = os.path.join(test_dir, "to_copy")
    os.mkdir(to_copy)

    source_file = pathlib.Path(to_copy) / "copy_file.txt"
    with open(source_file, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("dummy")

    entity_path = os.path.join(test_dir, "entity_name")
    os.mkdir(entity_path)

    dest_file = os.path.join(test_dir, "entity_name", "copy_file.txt")
    with open(dest_file, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("")

    parser = get_parser()
    cmd = f"copy {source_file} {dest_file}"
    args = cmd.split()
    ns = parser.parse_args(args)

    # Execute copy
    file_operations.copy(ns)

    # clean up
    os.remove(pathlib.Path(to_copy) / "copy_file.txt")
    os.rmdir(pathlib.Path(test_dir) / "to_copy")

    os.remove(pathlib.Path(entity_path) / "copy_file.txt")
    os.rmdir(pathlib.Path(test_dir) / "entity_name")


def test_copy_op_dirs(test_dir):
    """Test the oeprations that copies an entire directory tree source to a new location destination"""

    to_copy = os.path.join(test_dir, "to_copy")
    os.mkdir(to_copy)

    # write some test files in the dir
    source_file = pathlib.Path(to_copy) / "copy_file.txt"
    with open(source_file, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("dummy1")

    source_file_2 = pathlib.Path(to_copy) / "copy_file_2.txt"
    with open(source_file_2, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("dummy2")

    # entity_path to be the dest dir
    entity_path = os.path.join(test_dir, "entity_name")
    os.mkdir(entity_path)

    parser = get_parser()
    cmd = f"copy {to_copy} {entity_path}"
    args = cmd.split()
    ns = parser.parse_args(args)

    # Execute copy
    file_operations.copy(ns)

    # Clean up
    os.remove(pathlib.Path(to_copy) / "copy_file.txt")
    os.remove(pathlib.Path(to_copy) / "copy_file_2.txt")
    os.rmdir(pathlib.Path(test_dir) / "to_copy")
    os.remove(pathlib.Path(entity_path) / "copy_file.txt")
    os.remove(pathlib.Path(entity_path) / "copy_file_2.txt")
    os.rmdir(pathlib.Path(test_dir) / "entity_name")


def test_copy_op_bad_source_file(test_dir):
    """Test that a FileNotFoundError is raised when there is a bad source file"""

    to_copy = os.path.join(test_dir, "to_copy")
    os.mkdir(to_copy)
    entity_path = os.path.join(test_dir, "entity_name")
    os.mkdir(entity_path)

    bad_path = "/not/a/real/path"
    # Execute copy

    parser = get_parser()
    cmd = f"copy {bad_path} {entity_path}"
    args = cmd.split()
    ns = parser.parse_args(args)

    with pytest.raises(FileNotFoundError) as ex:
        file_operations.copy(ns)
    assert f"File or Directory {bad_path} not found" in ex.value.args[0]

    # clean up
    os.rmdir(pathlib.Path(test_dir) / "to_copy")
    os.rmdir(pathlib.Path(test_dir) / "entity_name")


def test_copy_op_bad_dest_path(test_dir):
    """Test that a FileNotFoundError is raised when there is a bad destination file."""

    to_copy = os.path.join(test_dir, "to_copy")
    os.mkdir(to_copy)

    source_file = pathlib.Path(to_copy) / "copy_file.txt"
    with open(source_file, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("dummy1")
    entity_path = os.path.join(test_dir, "entity_name")
    os.mkdir(entity_path)

    bad_path = "/not/a/real/path"

    parser = get_parser()
    cmd = f"copy {source_file} {bad_path}"
    args = cmd.split()
    ns = parser.parse_args(args)

    with pytest.raises(FileNotFoundError) as ex:
        file_operations.copy(ns)
    assert f"File or Directory {bad_path} not found" in ex.value.args[0]

    # clean up
    os.remove(pathlib.Path(to_copy) / "copy_file.txt")
    os.rmdir(pathlib.Path(test_dir) / "to_copy")
    os.rmdir(pathlib.Path(test_dir) / "entity_name")


def test_move_op(test_dir):
    """Test the operation to move a file"""

    source_dir = os.path.join(test_dir, "from_here")
    os.mkdir(source_dir)
    dest_dir = os.path.join(test_dir, "to_here")
    os.mkdir(dest_dir)

    # dest_file = os.path.join(test_output_root, "to_here", "to_here.txt")
    dest_file = pathlib.Path(dest_dir) / "to_here.txt"
    with open(dest_file, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write(" ")

    source_file = pathlib.Path(source_dir) / "app_move.txt"
    with open(source_file, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("dummy")

    assert osp.exists(source_file)
    with open(source_file, "r", encoding="utf-8") as dummy_file:
        assert dummy_file.read() == "dummy"

    parser = get_parser()
    cmd = f"move {source_file} {dest_file}"
    args = cmd.split()
    ns = parser.parse_args(args)

    file_operations.move(ns)

    # Assert that the move was successful
    assert not osp.exists(source_file)
    assert osp.exists(dest_file)
    with open(dest_file, "r", encoding="utf-8") as dummy_file:
        assert dummy_file.read() == "dummy"

    # Clean up the directories
    os.rmdir(source_dir)
    os.remove(dest_file)
    os.rmdir(dest_dir)


def test_remove_op(test_dir):
    """Test the operation to delete a file"""

    # Make a test file with dummy text
    to_del = pathlib.Path(test_dir) / "app_del.txt"
    with open(to_del, "w+", encoding="utf-8") as dummy_file:
        dummy_file.write("dummy")

    assert osp.exists(to_del)
    with open(to_del, "r", encoding="utf-8") as dummy_file:
        assert dummy_file.read() == "dummy"

    parser = get_parser()
    cmd = f"remove {to_del}"
    args = cmd.split()
    ns = parser.parse_args(args)

    file_operations.remove(ns)

    # Assert file has been deleted
    assert not osp.exists(to_del)


def test_remove_op_bad_path(test_dir):
    """Test that FileNotFoundError is raised when a bad path is given to the
    soperation to delete a file"""

    to_del = pathlib.Path(test_dir) / "not_real.txt"

    parser = get_parser()
    cmd = f"remove {to_del}"
    args = cmd.split()
    ns = parser.parse_args(args)

    with pytest.raises(FileNotFoundError) as ex:
        file_operations.remove(ns)
    assert f"File or Directory {to_del} not found" in ex.value.args[0]


@pytest.mark.parametrize(
    ["param_dict", "error_type"],
    [
        pytest.param(
            {
                "5": 10,
                "FIRST": "SECOND",
                "17": 20,
                "65": "70",
                "placeholder": "group leftupper region",
                "1200": "120",
            },
            "None",
            id="correct dict",
        ),
        pytest.param(
            ["list", "of", "values"],
            "TypeError",
            id="incorrect dict",
        ),
        pytest.param({}, "ValueError", id="empty dict"),
    ],
)
def test_configure_op(test_dir, fileutils, param_dict, error_type):
    """Test configure operation with correct parameter dictionary, empty dicitonary, and an incorrect type"""

    tag = ";"

    conf_path = fileutils.get_test_conf_path(osp.join("tagged_tests", "marked/"))
    # retrieve files to compare after test
    correct_path = fileutils.get_test_conf_path(osp.join("tagged_tests", "correct/"))

    # copy files to test directory
    dir_util.copy_tree(conf_path, test_dir)
    assert osp.isdir(test_dir)

    tagged_files = sorted(glob(test_dir + "/*"))
    correct_files = sorted(glob(correct_path + "*"))

    # Pickle the dictionary
    pickled_dict = pickle.dumps(param_dict)

    # Encode the pickled dictionary with Base64
    encoded_dict = base64.b64encode(pickled_dict)

    # Run configure op on test files
    for tagged_file in tagged_files:
        parser = get_parser()
        cmd = f"configure {tagged_file} {tagged_file} {tag} {encoded_dict}"
        args = cmd.split()
        ns = parser.parse_args(args)

        if error_type == "ValueError":
            with pytest.raises(ValueError) as ex:
                file_operations.configure(ns)
                assert "param dictionary is empty" in ex.value.args[0]
        elif error_type == "TypeError":
            with pytest.raises(TypeError) as ex:
                file_operations.configure(ns)
                assert "param dict is not a valid dictionary" in ex.value.args[0]
        else:
            file_operations.configure(ns)

    if error_type == "None":
        for written, correct in zip(tagged_files, correct_files):
            assert filecmp.cmp(written, correct)


def test_parser_move():
    """Test that the parser succeeds when receiving expected args for the move operation"""
    parser = get_parser()

    src_path = "/absolute/file/src/path"
    dest_path = "/absolute/file/dest/path"

    cmd = f"move {src_path} {dest_path}"
    args = cmd.split()
    ns = parser.parse_args(args)

    assert ns.source == src_path
    assert ns.dest == dest_path


def test_parser_remove():
    """Test that the parser succeeds when receiving expected args for the remove operation"""
    parser = get_parser()

    file_path = "/absolute/file/path"
    cmd = f"remove {file_path}"

    args = cmd.split()
    ns = parser.parse_args(args)

    assert ns.to_remove == file_path


def test_parser_symlink():
    """Test that the parser succeeds when receiving expected args for the symlink operation"""
    parser = get_parser()

    src_path = "/absolute/file/src/path"
    dest_path = "/absolute/file/dest/path"
    cmd = f"symlink {src_path} {dest_path}"

    args = cmd.split()

    ns = parser.parse_args(args)

    assert ns.source == src_path
    assert ns.dest == dest_path


def test_parser_copy():
    """Test that the parser succeeds when receiving expected args for the copy operation"""
    parser = get_parser()

    src_path = "/absolute/file/src/path"
    dest_path = "/absolute/file/dest/path"

    cmd = f"copy {src_path} {dest_path}"

    args = cmd.split()
    ns = parser.parse_args(args)

    assert ns.source == src_path
    assert ns.dest == dest_path


def test_parser_configure_parse():
    """Test that the parser succeeds when receiving expected args for the configure operation"""
    parser = get_parser()

    src_path = "/absolute/file/src/path"
    dest_path = "/absolute/file/dest/path"
    tag_delimiter = ";"

    param_dict = {
        "5": 10,
        "FIRST": "SECOND",
        "17": 20,
        "65": "70",
        "placeholder": "group leftupper region",
        "1200": "120",
    }

    pickled_dict = pickle.dumps(param_dict)
    encoded_dict = base64.b64encode(pickled_dict)

    cmd = f"configure {src_path} {dest_path} {tag_delimiter} {encoded_dict}"
    args = cmd.split()
    ns = parser.parse_args(args)

    assert ns.source == src_path
    assert ns.dest == dest_path
    assert ns.tag_delimiter == tag_delimiter
    assert ns.param_dict == str(encoded_dict)
