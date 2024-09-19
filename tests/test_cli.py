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

import argparse
import logging
import os
import pathlib
import typing as t
from contextlib import contextmanager

import pytest

import smartsim
from smartsim._core._cli import build, cli, plugin
from smartsim._core._cli.build import configure_parser as build_parser
from smartsim._core._cli.build import execute as build_execute
from smartsim._core._cli.clean import configure_parser as clean_parser
from smartsim._core._cli.clean import execute as clean_execute
from smartsim._core._cli.clean import execute_all as clobber_execute
from smartsim._core._cli.dbcli import execute as dbcli_execute
from smartsim._core._cli.site import execute as site_execute
from smartsim._core._cli.utils import MenuItemConfig
from smartsim._core._cli.validate import configure_parser as validate_parser
from smartsim._core._cli.validate import execute as validate_execute

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a

_TEST_LOGGER = logging.getLogger(__name__)

try:
    import smartdashboard
except:
    test_dash_plugin = False
else:
    test_dash_plugin = True


def mock_execute_custom(msg: str = None, good: bool = True) -> int:
    retval = 0 if good else 1
    print(msg)
    return retval


def mock_execute_good(
    _ns: argparse.Namespace, _unparsed: t.Optional[t.List[str]] = None
) -> int:
    return mock_execute_custom("GOOD THINGS", good=True)


def mock_execute_fail(
    _ns: argparse.Namespace, _unparsed: t.Optional[t.List[str]] = None
) -> int:
    return mock_execute_custom("BAD THINGS", good=False)


def test_cli_default_args_parsing(capsys):
    """Test default parser behaviors with no subparsers"""
    menu: t.List[cli.MenuItemConfig] = []
    smart_cli = cli.SmartCli(menu)

    captured = capsys.readouterr()  # throw away existing output

    with pytest.raises(SystemExit) as e:
        # the parser shouldn't get the `smart` CLI argument
        build_args = ["smart", "-h"]
        smart_cli.parser.parse_args(build_args)

    captured = capsys.readouterr()
    assert "invalid choice: 'smart'" in captured.err
    assert e.value.code == 2


def test_cli_invalid_command(capsys):
    """Ensure the response when an unsupported command is given"""
    exp_help = "this is my mock help text for build"
    exp_cmd = "build"
    actual_cmd = f"not{exp_cmd}"
    menu = [
        cli.MenuItemConfig(exp_cmd, exp_help, mock_execute_good, build.configure_parser)
    ]
    smart_cli = cli.SmartCli(menu)

    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as e:
        build_args = [actual_cmd, "-h"]
        smart_cli.parser.parse_args(build_args)

    captured = capsys.readouterr()  # capture new output

    # show that the command was not recognized
    assert "invalid choice" in captured.err
    assert e.value.code == 2


def test_cli_bad_default_args_parsing_bad_help(capsys):
    """Test passing an argument name that is incorrect"""
    menu: t.List[cli.MenuItemConfig] = []
    smart_cli = cli.SmartCli(menu)

    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as e:
        build_args = ["--halp"]  # <-- HELP vs HALP
        smart_cli.parser.parse_args(build_args)

    captured = capsys.readouterr()  # capture new output

    assert "smart: error:" in captured.err
    assert e.value.code == 2


def test_cli_bad_default_args_parsing_good_help(capsys):
    """Test passing an argument name that is correct"""
    menu: t.List[cli.MenuItemConfig] = []
    smart_cli = cli.SmartCli(menu)

    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as e:
        build_args = ["-h"]
        smart_cli.parser.parse_args(build_args)

    captured = capsys.readouterr()  # capture new output

    assert "smart: error:" not in captured.out
    assert "usage: smart" in captured.out
    assert e.value.code == 0


def test_cli_add_subparser(capsys):
    """Test that passing configuration for a command causes the command
    to be added to the CLI
    """
    exp_help = "this is my mock help text for build"
    exp_cmd = "build"
    menu = [
        cli.MenuItemConfig(exp_cmd, exp_help, mock_execute_good, build.configure_parser)
    ]
    smart_cli = cli.SmartCli(menu)

    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as e:
        build_args = [exp_cmd, "-h"]  # <--- -h only
        smart_cli.parser.parse_args(build_args)

    captured = capsys.readouterr()  # capture new output

    # show that -h showed the expected help text
    assert exp_help in captured.out
    assert e.value.code == 0

    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as e:
        build_args = [exp_cmd, "--help"]
        smart_cli.parser.parse_args(build_args)

    captured = capsys.readouterr()  # capture new output

    # show that --help ALSO works
    assert exp_help in captured.out
    assert e.value.code == 0


def test_cli_subparser_selection(capsys):
    """Ensure the right subparser is selected"""
    exp_a_help = "this is my mock help text for dbcli"
    exp_a_cmd = "dbcli"

    exp_b_help = "this is my mock help text for build"
    exp_b_cmd = "build"

    menu = [
        cli.MenuItemConfig(
            exp_a_cmd, exp_a_help, mock_execute_good, build.configure_parser
        ),
        cli.MenuItemConfig(
            exp_b_cmd, exp_b_help, mock_execute_good, build.configure_parser
        ),
    ]
    smart_cli = cli.SmartCli(menu)

    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as e:
        build_args = [exp_a_cmd, "-h"]  # <--- -h only
        smart_cli.parser.parse_args(build_args)

    captured = capsys.readouterr()  # capture new output

    # show that -h showed the expected help text for `smart dbcli -h`
    assert exp_a_help in captured.out
    assert e.value.code == 0

    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as e:
        build_args = [exp_b_cmd, "--help"]
        smart_cli.parser.parse_args(build_args)

    captured = capsys.readouterr()  # capture new output

    # show that -h showed the expected help text for `smart build -h`
    assert exp_b_help in captured.out
    assert e.value.code == 0


def test_cli_command_execution(capsys):
    """Ensure the right command is executed"""
    exp_a_help = "this is my mock help text for dbcli"
    exp_a_cmd = "dbcli"

    exp_b_help = "this is my mock help text for build"
    exp_b_cmd = "build"

    dbcli_exec = lambda x, y: mock_execute_custom(msg="Database", good=True)
    build_exec = lambda x, y: mock_execute_custom(msg="Builder", good=True)

    menu = [
        cli.MenuItemConfig(exp_a_cmd, exp_a_help, dbcli_exec, lambda x: None),
        cli.MenuItemConfig(exp_b_cmd, exp_b_help, build_exec, lambda x: None),
    ]
    smart_cli = cli.SmartCli(menu)

    captured = capsys.readouterr()  # throw away existing output

    build_args = ["smart", exp_a_cmd]
    ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr()  # capture new output

    # show that `smart dbcli` calls the build parser and build execute function
    assert "Database" in captured.out
    assert ret_val == 0

    build_args = ["smart", exp_b_cmd]
    ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr()  # capture new output

    # show that `smart build` calls the build parser and build execute function
    assert "Builder" in captured.out
    assert ret_val == 0


def test_cli_default_cli(capsys):
    """Ensure the default CLI supports expected top-level commands"""
    smart_cli = cli.default_cli()

    captured = capsys.readouterr()  # throw away existing output

    # execute with no <command> argument, expect full help text
    build_args = ["smart"]
    ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr()  # capture new output

    # show that `smart dbcli` calls the build parser and build execute function
    assert "usage: smart [-h] <command>" in captured.out
    assert "Available commands" in captured.out
    assert ret_val == os.EX_USAGE

    # execute with `build` argument, expect build-specific help text
    with pytest.raises(SystemExit) as e:
        build_args = ["smart", "build", "-h"]
        ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr()  # capture new output

    assert "usage: smart build [-h]" in captured.out
    assert "Build SmartSim dependencies" in captured.out
    assert "optional arguments:" in captured.out or "options:" in captured.out
    assert ret_val == os.EX_USAGE

    # execute with `clean` argument, expect clean-specific help text
    with pytest.raises(SystemExit) as e:
        build_args = ["smart", "clean", "-h"]
        ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr()  # capture new output

    assert "usage: smart clean [-h]" in captured.out
    assert "Remove previous ML runtime installation" in captured.out
    assert "optional arguments:" in captured.out or "options:" in captured.out
    assert "--clobber" in captured.out
    assert ret_val == os.EX_USAGE

    # execute with `dbcli` argument, expect dbcli-specific help text
    with pytest.raises(SystemExit) as e:
        build_args = ["smart", "dbcli", "-h"]
        ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr()  # capture new output

    assert "usage: smart dbcli [-h]" in captured.out
    assert "Print the path to the redis-cli binary" in captured.out
    assert "optional arguments:" in captured.out or "options:" in captured.out
    assert ret_val == os.EX_USAGE

    # execute with `site` argument, expect site-specific help text
    with pytest.raises(SystemExit) as e:
        build_args = ["smart", "site", "-h"]
        ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr()  # capture new output

    assert "usage: smart site [-h]" in captured.out
    assert "Print the installation site of SmartSim" in captured.out
    assert "optional arguments:" in captured.out or "options:" in captured.out
    assert ret_val == os.EX_USAGE

    # execute with `clobber` argument, expect clobber-specific help text
    with pytest.raises(SystemExit) as e:
        build_args = ["smart", "clobber", "-h"]
        ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr()  # capture new output

    assert "usage: smart clobber [-h]" in captured.out
    assert "Remove all previous dependency installations" in captured.out
    assert "optional arguments:" in captured.out or "options:" in captured.out
    # assert "--clobber" not in captured.out
    assert ret_val == os.EX_USAGE


@pytest.mark.skipif(not test_dash_plugin, reason="plugin not found")
def test_cli_plugin_dashboard(capfd):
    """Ensure expected dashboard CLI plugin commands are supported"""
    smart_cli = cli.default_cli()
    capfd.readouterr()  # throw away existing output

    # execute with `dashboard` argument, expect dashboard-specific help text
    build_args = ["smart", "dashboard", "-h"]
    rc = smart_cli.execute(build_args)

    captured = capfd.readouterr()  # capture new output

    assert "[-d DIRECTORY]" in captured.out
    assert "[-p PORT]" in captured.out

    assert "optional arguments:" in captured.out
    assert rc == 0


def test_cli_plugin_invalid(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Ensure unexpected CLI plugins are reported"""
    import smartsim._core._cli.cli
    import smartsim._core._cli.plugin

    plugin_module = "notinstalled.Experiment_Overview"
    bad_plugins = [
        lambda: MenuItemConfig(
            "dashboard",
            "Start the SmartSim dashboard",
            plugin.dynamic_execute(plugin_module, "Dashboard!"),
            is_plugin=True,
        )
    ]
    monkeypatch.setattr(smartsim._core._cli.cli, "plugins", bad_plugins)
    # Coloredlogs doesn't play nice with caplog
    monkeypatch.setattr(
        smartsim._core._cli.plugin,
        "_LOGGER",
        _TEST_LOGGER,
    )

    smart_cli = cli.default_cli()

    # execute with `dashboard` argument, expect failure to find dashboard plugin
    build_args = ["smart", "dashboard", "-h"]

    rc = smart_cli.execute(build_args)

    assert plugin_module in caplog.text
    assert "not found" in caplog.text
    assert rc == os.EX_CONFIG


# fmt: off
@pytest.mark.parametrize(
    "command,mock_location,exp_output",
    [
        pytest.param("build", "build_execute", "mocked-build", id="ensure build action is executed"),
        pytest.param("clean", "clean_execute", "mocked-clean", id="ensure clean action is executed"),
        pytest.param("dbcli", "dbcli_execute", "mocked-dbcli", id="ensure dbcli action is executed"),
        pytest.param("site", "site_execute", "mocked-site", id="ensure site action is executed"),
        pytest.param("clobber", "clobber_execute", "mocked-clobber", id="ensure clobber action is executed"),
        pytest.param("validate", "validate_execute", "mocked-validate", id="ensure validate action is executed"),
        pytest.param("info", "info_execute", "mocked-validate", id="ensure info action is executed"),
    ]
)
# fmt: on
def test_cli_action(capsys, monkeypatch, command, mock_location, exp_output):
    """Ensure the default CLI executes the build action"""

    def mock_execute(ns: argparse.Namespace, _unparsed: t.Optional[t.List[str]] = None):
        print(exp_output)
        return 0

    monkeypatch.setattr(smartsim._core._cli.cli, mock_location, mock_execute)

    smart_cli = cli.default_cli()

    captured = capsys.readouterr()  # throw away existing output

    # execute with `<command>` argument, expect <command>-specific output text
    build_args = ["smart", command]
    ret_val = smart_cli.execute(build_args)

    captured = capsys.readouterr()  # capture new output

    assert exp_output in captured.out
    assert ret_val == 0


# fmt: off
@pytest.mark.parametrize(
                       "command,      mock_location,                    exp_output,            optional_arg, exp_valid,                    exp_err_msg,  check_prop, exp_prop_val",
    [
        pytest.param(   "build",    "build_execute",        "verbose mocked-build",                    "-v",      True,                             "",         "v",         True, id="verbose 'on'"),
        pytest.param(   "build",    "build_execute",            "cpu mocked-build",          "--device=cpu",      True,                             "",    "device",        "cpu", id="device 'cpu'"),
        pytest.param(   "build",    "build_execute",           "gpuX mocked-build",         "--device=gpux",     False,       "invalid choice: 'gpux'",          "",           "", id="set bad device 'gpuX'"),
        pytest.param(   "build",    "build_execute",  "no tensorflow mocked-build",     "--skip-tensorflow",      True,                             "",     "no_tf",         True, id="Skip TF"),
        pytest.param(   "build",    "build_execute",       "no torch mocked-build",          "--skip-torch",      True,                             "",     "no_pt",         True, id="Skip Torch"),
        pytest.param(   "build",    "build_execute",           "onnx mocked-build",           "--skip-onnx",      True,                             "",      "onnx",         True, id="Skip Onnx"),
        pytest.param(   "build",    "build_execute",     "config-dir mocked-build", "--config-dir /foo/bar",      True,                             "", "config-dir",   "/foo/bar", id="set torch dir"),
        pytest.param(   "build",    "build_execute", "bad-config-dir mocked-build",          "--config-dir",     False, "error: argument --config-dir",          "",           "", id="set config dir w/o path"),
        pytest.param(   "build",    "build_execute",          "keydb mocked-build",               "--keydb",      True,                             "",     "keydb",         True, id="keydb on"),
        pytest.param(   "clean",    "clean_execute",     "clobbering mocked-clean",             "--clobber",      True,                             "",   "clobber",         True, id="clean w/clobber"),
        pytest.param("validate", "validate_execute",        "port mocked-validate",          "--port=12345",      True,                             "",      "port",        12345, id="validate w/ manual port"),
        pytest.param("validate", "validate_execute",  "abbrv port mocked-validate",              "-p 12345",      True,                             "",      "port",        12345, id="validate w/ manual abbreviated port"),
        pytest.param("validate", "validate_execute",         "cpu mocked-validate",          "--device=cpu",      True,                             "",    "device",        "cpu", id="validate: device 'cpu'"),
        pytest.param("validate", "validate_execute",         "gpu mocked-validate",          "--device=gpu",      True,                             "",    "device",        "gpu", id="validate: device 'gpu'"),
        pytest.param("validate", "validate_execute",        "gpuX mocked-validate",         "--device=gpux",     False,       "invalid choice: 'gpux'",          "",           "", id="validate: set bad device 'gpuX'"),
    ]
)
# fmt: on
def test_cli_optional_args(
    capsys,
    monkeypatch,
    command: str,
    mock_location: str,
    exp_output: str,
    optional_arg: str,
    exp_valid: bool,
    exp_err_msg: str,
    check_prop: str,
    exp_prop_val: t.Any,
):
    """Ensure the parser for a command handles expected optional arguments"""

    def mock_execute(ns: argparse.Namespace, _unparsed: t.Optional[t.List[str]] = None):
        print(exp_output)
        return 0

    monkeypatch.setattr(smartsim._core._cli.cli, mock_location, mock_execute)

    smart_cli = cli.default_cli()

    captured = capsys.readouterr()  # throw away existing output

    build_args = ["smart", command] + optional_arg.split()
    if exp_valid:
        ret_val = smart_cli.execute(build_args)

        captured = capsys.readouterr()  # capture new output

        assert exp_output in captured.out  # did the expected execution method occur?
        assert ret_val == 0  # is the retval is non-failure code?
    else:
        with pytest.raises(SystemExit) as e:
            ret_val = smart_cli.execute(build_args)
            assert ret_val > 0

        captured = capsys.readouterr()  # capture new output
        assert exp_err_msg in captured.err


# fmt: off
@pytest.mark.parametrize(
    "command,mock_location,mock_output,exp_output",
    [
        pytest.param("build", "build_execute", "verbose mocked-build", "usage: smart build", id="build"),
        pytest.param("clean", "clean_execute", "helpful mocked-clean", "usage: smart clean", id="clean"),
        pytest.param("clobber", "clean_execute", "helpful mocked-clobber", "usage: smart clobber", id="clobber"),
        pytest.param("dbcli", "clean_execute", "helpful mocked-dbcli", "usage: smart dbcli", id="dbcli"),
        pytest.param("site", "clean_execute", "helpful mocked-site", "usage: smart site", id="site"),
        pytest.param("validate", "validate_execute", "helpful mocked-validate", "usage: smart validate", id="validate"),
        pytest.param("info", "info_execute", "helpful mocked-validate", "usage: smart info", id="info"),
    ]
)
# fmt: on
def test_cli_help_support(
    capsys,
    monkeypatch,
    command: str,
    mock_location: str,
    mock_output: str,
    exp_output: str,
):
    """Ensure the parser supports help optional for commands as expected"""

    def mock_execute(ns: argparse.Namespace, unparsed: t.Optional[t.List[str]] = None):
        print(mock_output)
        return 0

    monkeypatch.setattr(smartsim._core._cli.cli, mock_location, mock_execute)

    smart_cli = cli.default_cli()

    captured = capsys.readouterr()  # throw away existing output

    # execute with `<command>` argument, expect <command>-specific help text
    build_args = ["smart", command] + ["-h"]
    with pytest.raises(SystemExit) as e:
        ret_val = smart_cli.execute(build_args)
        assert ret_val == 0

    captured = capsys.readouterr()  # capture new output
    assert exp_output in captured.out


# fmt: off
@pytest.mark.parametrize(
    "command,mock_location,exp_output",
    [
        pytest.param("build", "build_execute", "verbose mocked-build", id="build"),
        pytest.param("clean", "clean_execute", "verbose mocked-clean", id="clean"),
        pytest.param("clobber", "clobber_execute", "verbose mocked-clobber", id="clobber"),
        pytest.param("dbcli", "dbcli_execute", "verbose mocked-dbcli", id="dbcli"),
        pytest.param("site", "site_execute", "verbose mocked-site", id="site"),
        pytest.param("validate", "validate_execute", "verbose mocked-validate", id="validate"),
        pytest.param("info", "info_execute", "verbose mocked-validate", id="validate"),
    ]
)
# fmt: on
def test_cli_invalid_optional_args(
    capsys, monkeypatch, command: str, mock_location: str, exp_output: str
):
    """Ensure the parser throws expected error for an invalid argument"""

    def mock_execute(ns: argparse.Namespace, unparsed: t.Optional[t.List[str]] = None):
        print(exp_output)
        return 0

    monkeypatch.setattr(smartsim._core._cli.cli, mock_location, mock_execute)

    smart_cli = cli.default_cli()

    captured = capsys.readouterr()  # throw away existing output

    # execute with `<command>` argument, expect CLI to raise invalid arg error
    build_args = ["smart", command] + ["-xyz"]
    with pytest.raises(SystemExit) as e:
        ret_val = smart_cli.execute(build_args)
        assert ret_val > 0

    captured = capsys.readouterr()  # capture new output
    assert "unrecognized argument" in captured.err


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("build", id="build"),
        pytest.param("clean", id="clean"),
        pytest.param("clobber", id="clobber"),
        pytest.param("dbcli", id="dbcli"),
        pytest.param("site", id="site"),
        pytest.param("validate", id="validate"),
        pytest.param("info", id="info"),
    ],
)
def test_cli_invalid_optional_args(capsys, command):
    """Ensure the parser throws expected error for an invalid command"""
    smart_cli = cli.default_cli()

    captured = capsys.readouterr()  # throw away existing output

    # execute with `<command>` argument, expect CLI to raise invalid arg error
    build_args = ["smart", command] + ["-xyz"]
    with pytest.raises(SystemExit) as e:
        ret_val = smart_cli.execute(build_args)
        assert ret_val > 0

    captured = capsys.readouterr()  # capture new output
    assert "unrecognized argument" in captured.err


def test_cli_full_clean_execute(capsys, monkeypatch):
    """Ensure that the execute method of clean is called"""
    exp_retval = 0
    exp_output = "mocked-clean utility"

    # mock out the internal clean method so we don't actually delete anything
    def mock_clean(core_path: pathlib.Path, _all: bool = False) -> int:
        print(exp_output)
        return exp_retval

    monkeypatch.setattr(smartsim._core._cli.clean, "clean", mock_clean)

    command = "clean"
    cfg = MenuItemConfig(
        command, f"test {command} help text", clean_execute, clean_parser
    )
    menu = [cfg]
    smart_cli = cli.SmartCli(menu)

    captured = capsys.readouterr()  # throw away existing output

    build_args = ["smart", command]
    actual_retval = smart_cli.execute(build_args)

    captured = capsys.readouterr()  # capture new output

    assert exp_output in captured.out
    assert actual_retval == exp_retval


def test_cli_full_clobber_execute(capsys, monkeypatch):
    """Ensure that the execute method of clobber is called"""
    exp_retval = 0
    exp_output = "mocked-clobber utility"

    def mock_operation(*args, **kwargs) -> int:
        print(exp_output)
        return exp_retval

    # mock out the internal clean method so we don't actually delete anything
    monkeypatch.setattr(smartsim._core._cli.clean, "clean", mock_operation)

    command = "clobber"
    cfg = MenuItemConfig(command, f"test {command} help text", clobber_execute)
    menu = [cfg]
    smart_cli = cli.SmartCli(menu)

    captured = capsys.readouterr()  # throw away existing output

    build_args = ["smart", command]
    actual_retval = smart_cli.execute(build_args)

    captured = capsys.readouterr()  # capture new output

    assert exp_output in captured.out
    assert actual_retval == exp_retval


def test_cli_full_dbcli_execute(capsys, monkeypatch):
    """Ensure that the execute method of dbcli is called"""
    exp_retval = 0
    exp_output = "mocked-get_db_path utility"

    def mock_operation(*args, **kwargs) -> int:
        return exp_output

    # mock out the internal get_db_path method so we don't actually do file system ops
    monkeypatch.setattr(smartsim._core._cli.dbcli, "get_db_path", mock_operation)

    command = "dbcli"
    cfg = MenuItemConfig(command, f"test {command} help text", dbcli_execute)
    menu = [cfg]
    smart_cli = cli.SmartCli(menu)

    captured = capsys.readouterr()  # throw away existing output

    build_args = ["smart", command]
    actual_retval = smart_cli.execute(build_args)

    captured = capsys.readouterr()  # capture new output

    assert exp_output in captured.out
    assert actual_retval == exp_retval


def test_cli_full_site_execute(capsys, monkeypatch):
    """Ensure that the execute method of site is called"""
    exp_retval = 0
    exp_output = "mocked-get_install_path utility"

    def mock_operation(*args, **kwargs) -> int:
        print(exp_output)
        return exp_retval

    # mock out the internal get_db_path method so we don't actually do file system ops
    monkeypatch.setattr(smartsim._core._cli.site, "get_install_path", mock_operation)

    command = "site"
    cfg = MenuItemConfig(command, f"test {command} help text", site_execute)
    menu = [cfg]
    smart_cli = cli.SmartCli(menu)

    captured = capsys.readouterr()  # throw away existing output

    build_args = ["smart", command]
    actual_retval = smart_cli.execute(build_args)

    captured = capsys.readouterr()  # capture new output

    assert exp_output in captured.out
    assert actual_retval == exp_retval


def test_cli_full_build_execute(capsys, monkeypatch):
    """Ensure that the execute method of build is called"""
    exp_retval = 0
    exp_output = "mocked-execute-build utility"

    def mock_operation(*args, **kwargs) -> int:
        print(exp_output)
        return exp_retval

    # mock out the internal get_db_path method so we don't actually do file system ops
    monkeypatch.setattr(smartsim._core._cli.build, "tabulate", mock_operation)
    monkeypatch.setattr(smartsim._core._cli.build, "build_database", mock_operation)
    monkeypatch.setattr(smartsim._core._cli.build, "build_redis_ai", mock_operation)

    command = "build"
    cfg = MenuItemConfig(
        command, f"test {command} help text", build_execute, build_parser
    )
    menu = [cfg]
    smart_cli = cli.SmartCli(menu)

    captured = capsys.readouterr()  # throw away existing output

    build_args = ["smart", command]
    actual_retval = smart_cli.execute(build_args)

    captured = capsys.readouterr()  # capture new output

    assert exp_output in captured.out
    assert actual_retval == exp_retval


def _good_build(*args, **kwargs):
    _TEST_LOGGER.info("LGTM")


def _bad_build(*args, **kwargs):
    raise Exception


@contextmanager
def _mock_temp_dir(*a, **kw):
    yield "/a/mock/path/to/a/mock/temp/dir"


@pytest.mark.parametrize(
    "mock_verify_fn, expected_stdout, expected_retval",
    [
        pytest.param(_good_build, "LGTM", os.EX_OK, id="Configured Correctly"),
        pytest.param(
            _bad_build,
            "SmartSim failed to run a simple experiment",
            os.EX_SOFTWARE,
            id="Configured Incorrectly",
        ),
    ],
)
def test_cli_validation_test_execute(
    caplog,
    monkeypatch,
    mock_verify_fn,
    expected_stdout,
    expected_retval,
):
    """Ensure the that the execute method of test target is called. This test will
    stub out the actual test run by the cli (it will be tested elsewere), and simply
    checks that if at any point the test raises an exception an appropriate error
    code and error msg are returned.
    """
    caplog.set_level(logging.INFO)

    # Mock out the verification tests/avoid file system ops
    monkeypatch.setattr(smartsim._core._cli.validate, "test_install", mock_verify_fn)
    monkeypatch.setattr(
        smartsim._core._cli.validate,
        "_VerificationTempDir",
        _mock_temp_dir,
    )
    # Coloredlogs doesn't play nice with caplog
    monkeypatch.setattr(
        smartsim._core._cli.validate,
        "logger",
        _TEST_LOGGER,
    )

    command = "validate"
    cfg = MenuItemConfig(
        command, f"test {command} help text", validate_execute, validate_parser
    )
    menu = [cfg]
    smart_cli = cli.SmartCli(menu)

    verify_args = ["smart", command]
    actual_retval = smart_cli.execute(verify_args)

    assert expected_stdout in caplog.text
    assert actual_retval == expected_retval


def test_validate_correctly_sets_and_restores_env(monkeypatch):
    monkeypatch.setenv("FOO", "BAR")
    monkeypatch.setenv("SPAM", "EGGS")
    monkeypatch.delenv("TICK", raising=False)
    monkeypatch.delenv("DNE", raising=False)

    assert os.environ["FOO"] == "BAR"
    assert os.environ["SPAM"] == "EGGS"
    assert "TICK" not in os.environ
    assert "DNE" not in os.environ

    with smartsim._core._cli.validate._env_vars_set_to(
        {
            "FOO": "BAZ",  # Redefine
            "SPAM": None,  # Delete
            "TICK": "TOCK",  # Add
            "DNE": None,  # Delete already missing
        }
    ):
        assert os.environ["FOO"] == "BAZ"
        assert "SPAM" not in os.environ
        assert os.environ["TICK"] == "TOCK"
        assert "DNE" not in os.environ

    assert os.environ["FOO"] == "BAR"
    assert os.environ["SPAM"] == "EGGS"
    assert "TICK" not in os.environ
    assert "DNE" not in os.environ
