# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

import argparse
import pytest
import smartsim
import typing as t
from smartsim._core._cli import utils, site, dbcli, cli, clean, build


def mock_execute_custom(msg: str = None, good: bool = True) -> int:
    retval = 0 if good else 1
    print(msg)
    return retval

def mock_execute_good(ns: argparse.Namespace) -> int:
    return mock_execute_custom("GOOD THINGS", good = True)


def mock_execute_fail(ns: argparse.Namespace) -> int:
    return mock_execute_custom("BAD THINGS", good = False)


def test_default_args_parsing(capsys):
    """Test default parser behaviors with no subparsers"""
    menu: t.List[cli.MenuItemConfig] = []
    smart_cli = cli.SmartCli(menu)    
    
    captured = capsys.readouterr()  # throw away existing output

    with pytest.raises(SystemExit) as e:    
        # the parser shouldn't get the `smart` CLI argument
        build_args = ["smart", "-h"]
        smart_cli.parser.parse_args(build_args)

    captured = capsys.readouterr()
    assert "invalid choice: \'smart\'" in captured.err
    assert e.value.code == 2


def test_bad_default_args_parsing_bad_help(capsys):
    """Test passing an argument name that is incorrect"""
    menu: t.List[cli.MenuItemConfig] = []
    smart_cli = cli.SmartCli(menu)    
    
    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as e:
        build_args = ["--halp"]  # <-- HELP vs HALP
        smart_cli.parser.parse_args(build_args)
        
    captured = capsys.readouterr() # capture new output
    
    assert "smart: error:" in captured.err
    assert e.value.code == 2


def test_bad_default_args_parsing_good_help(capsys):
    """Test passing an argument name that is correct"""
    menu: t.List[cli.MenuItemConfig] = []
    smart_cli = cli.SmartCli(menu)    
    
    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as e:
        build_args = ["-h"]
        smart_cli.parser.parse_args(build_args)
        
    captured = capsys.readouterr() # capture new output
    
    assert "smart: error:" not in captured.out
    assert "usage: smart" in captured.out
    assert e.value.code == 0


def test_add_subparser(capsys):
    """Test passing a subparser command"""
    exp_help = "this is my mock help text for build"
    exp_cmd = "build"
    menu = [cli.MenuItemConfig(exp_cmd,
                               exp_help,
                               mock_execute_good,
                               build.configure_parser)]
    smart_cli = cli.SmartCli(menu)    
    
    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as e:
        build_args = [exp_cmd, "-h"]      # <--- -h only    
        smart_cli.parser.parse_args(build_args)
        
    captured = capsys.readouterr() # capture new output
    
    # show that -h showed the expected help text
    assert exp_help in captured.out
    assert e.value.code == 0

    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as e:
        build_args = [exp_cmd, "--help"]
        smart_cli.parser.parse_args(build_args)
        
    captured = capsys.readouterr() # capture new output
    
    # show that --help ALSO works
    assert exp_help in captured.out
    assert e.value.code == 0


def test_subparser_selection(capsys):
    """Ensure the right subparser is selected"""
    exp_a_help = "this is my mock help text for dbcli"
    exp_a_cmd = "dbcli"

    exp_b_help = "this is my mock help text for build"
    exp_b_cmd = "build"

    menu = [cli.MenuItemConfig(exp_a_cmd,
                               exp_a_help,
                               mock_execute_good,
                               build.configure_parser),
            cli.MenuItemConfig(exp_b_cmd,
                               exp_b_help,
                               mock_execute_good,
                               build.configure_parser)]
    smart_cli = cli.SmartCli(menu)    
    
    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as e:
        build_args = [exp_a_cmd, "-h"]      # <--- -h only    
        smart_cli.parser.parse_args(build_args)
        
    captured = capsys.readouterr() # capture new output
    
    # show that -h showed the expected help text for `smart dbcli -h`
    assert exp_a_help in captured.out
    assert e.value.code == 0

    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as e:
        build_args = [exp_b_cmd, "--help"]
        smart_cli.parser.parse_args(build_args)
        
    captured = capsys.readouterr() # capture new output
    
    # show that -h showed the expected help text for `smart build -h`
    assert exp_b_help in captured.out
    assert e.value.code == 0


def test_command_execution(capsys):
    """Ensure the right command is executed"""
    exp_a_help = "this is my mock help text for dbcli"
    exp_a_cmd = "dbcli"

    exp_b_help = "this is my mock help text for build"
    exp_b_cmd = "build"
    
    dbcli_exec = lambda x: mock_execute_custom(msg="Database", good=True)
    build_exec = lambda x: mock_execute_custom(msg="Builder", good=True)
    
    menu = [cli.MenuItemConfig(exp_a_cmd,
                               exp_a_help,
                               dbcli_exec,
                               lambda x: None),
            cli.MenuItemConfig(exp_b_cmd,
                               exp_b_help,
                               build_exec,
                               lambda x: None)]
    smart_cli = cli.SmartCli(menu)    
    
    captured = capsys.readouterr()  # throw away existing output
    
    build_args = ["smart", exp_a_cmd]
    ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr() # capture new output
    
    # show that `smart dbcli` calls the build parser and build execute function
    assert "Database" in captured.out
    assert ret_val == 0

    build_args = ["smart", exp_b_cmd]
    ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr() # capture new output
    
    # show that `smart build` calls the build parser and build execute function
    assert "Builder" in captured.out
    assert ret_val == 0


def test_default_cli(capsys):
    """Ensure the default CLI supports expected top-level commands"""
    smart_cli = cli.default_cli()
    
    captured = capsys.readouterr()  # throw away existing output
    
    # execute with no <command> argument, expect full help text
    build_args = ["smart"]
    ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr() # capture new output
    
    # show that `smart dbcli` calls the build parser and build execute function
    assert "usage: smart [-h] <command>" in captured.out
    assert "Available commands" in captured.out
    assert ret_val == 0

    # execute with `build` argument, expect build-specific help text
    with pytest.raises(SystemExit) as e:
        build_args = ["smart", "build", "-h"]
        ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr() # capture new output
    
    assert "usage: smart build [-h]" in captured.out
    assert "Build SmartSim dependencies" in captured.out
    assert "optional arguments" in captured.out
    assert ret_val == 0

    # execute with `clean` argument, expect clean-specific help text
    with pytest.raises(SystemExit) as e:
        build_args = ["smart", "clean", "-h"]
        ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr() # capture new output
    
    assert "usage: smart clean [-h]" in captured.out
    assert "Remove previous ML runtime installation" in captured.out
    assert "optional arguments" in captured.out
    assert "--clobber" in captured.out
    assert ret_val == 0

    # execute with `dbcli` argument, expect dbcli-specific help text
    with pytest.raises(SystemExit) as e:
        build_args = ["smart", "dbcli", "-h"]
        ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr() # capture new output
    
    assert "usage: smart dbcli [-h]" in captured.out
    assert "Print the path to the redis-cli binary" in captured.out
    assert "optional arguments" in captured.out
    assert ret_val == 0

    # execute with `site` argument, expect site-specific help text
    with pytest.raises(SystemExit) as e:
        build_args = ["smart", "site", "-h"]
        ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr() # capture new output
    
    assert "usage: smart site [-h]" in captured.out
    assert "Print the installation site of SmartSim" in captured.out
    assert "optional arguments" in captured.out
    assert ret_val == 0

    # execute with `clobber` argument, expect clobber-specific help text
    with pytest.raises(SystemExit) as e:
        build_args = ["smart", "clobber", "-h"]
        ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr() # capture new output
    
    assert "usage: smart clobber [-h]" in captured.out
    assert "Remove all previous dependency installations" in captured.out
    assert "optional arguments" in captured.out
    # assert "--clobber" not in captured.out
    assert ret_val == 0


@pytest.mark.parametrize(
    "command,mock_location,exp_output",
    [
        pytest.param("build", "build_execute", "mocked-build", id="ensure build action is executed"),
        pytest.param("clean", "clean_execute", "mocked-clean", id="ensure clean action is executed"),
        pytest.param("dbcli", "dbcli_execute", "mocked-dbcli", id="ensure dbcli action is executed"),
        pytest.param("site", "site_execute", "mocked-site", id="ensure site action is executed"),
        pytest.param("clobber", "clobber_execute", "mocked-clobber", id="ensure clobber action is executed"),
    ]
)
def test_cli_action(capsys, monkeypatch, command, mock_location, exp_output):
    """Ensure the default CLI executes the build action"""
    def mock_execute(ns: argparse.Namespace):
        print(exp_output)
        return 0

    monkeypatch.setattr(smartsim._core._cli.cli, mock_location, mock_execute)
    
    smart_cli = cli.default_cli()
    
    captured = capsys.readouterr()  # throw away existing output
    
    # execute with `<command>` argument, expect <command>-specific help text
    build_args = ["smart", command]
    ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr() # capture new output
    
    assert exp_output in captured.out
    assert ret_val == 0


@pytest.mark.parametrize(
    "command,mock_location,exp_output,optional_arg,exp_valid,exp_err_msg",
    [
        pytest.param("build", "build_execute", "verbose mocked-build", "-v", True, "", id="verbose 'on'"),
        pytest.param("build", "build_execute", "cpu mocked-build", "--device=cpu", True, "", id="device 'cpu'"),
        pytest.param("build", "build_execute", "gpu mocked-build", "--device=gpu", True, "", id="device 'gpu'"),
        pytest.param("build", "build_execute", "gpuX mocked-build", "--device=gpux", False, "invalid choice: 'gpux'", id="set bad device 'gpuX'"),
        pytest.param("build", "build_execute", "no tensorflow mocked-build", "--no_pt", True, "", id="set no TF"),
        pytest.param("build", "build_execute", "no torch mocked-build", "--no_pt", True, "", id="set no torch"),
        pytest.param("build", "build_execute", "onnx mocked-build", "--onnx", True, "", id="set w/onnx"),
        pytest.param("build", "build_execute", "torch-dir mocked-build", "--torch_dir /foo/bar", True, "", id="set torch dir"),
        pytest.param("build", "build_execute", "bad-torch-dir mocked-build", "--torch_dir", False, "error: argument --torch_dir", id="set torch dir, no path"),
        pytest.param("build", "build_execute", "keydb mocked-build", "--keydb", True, "", id="keydb on"),
        pytest.param("build", "build_execute", "only-pkg mocked-build", "--only_python_packages", True, "", id="only-python-packages on"),
    ]
)
def test_cli_optional_args(capsys, 
                           monkeypatch, 
                           command, 
                           mock_location, 
                           exp_output, 
                           optional_arg, 
                           exp_valid,
                           exp_err_msg):
    """Ensure the parser for a command handles expected optional arguments"""
    def mock_execute(ns: argparse.Namespace):
        print(exp_output)
        return 0

    monkeypatch.setattr(smartsim._core._cli.cli, mock_location, mock_execute)
    
    smart_cli = cli.default_cli()
    
    captured = capsys.readouterr()  # throw away existing output
    
    # execute with `<command>` argument, expect <command>-specific help text
    build_args = ["smart", command] + optional_arg.split()
    if exp_valid:
        ret_val = smart_cli.execute(build_args)

        captured = capsys.readouterr() # capture new output
        
        assert exp_output in captured.out
        assert ret_val == 0
    else:
        with pytest.raises(SystemExit) as e:
            ret_val = smart_cli.execute(build_args)
            assert ret_val > 0

        captured = capsys.readouterr() # capture new output
        assert exp_err_msg in captured.err
