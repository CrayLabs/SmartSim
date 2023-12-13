#!/usr/bin/env python

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
import os
import typing as t

from smartsim._core._cli.build import configure_parser as build_parser
from smartsim._core._cli.build import execute as build_execute
from smartsim._core._cli.clean import configure_parser as clean_parser
from smartsim._core._cli.clean import execute as clean_execute
from smartsim._core._cli.clean import execute_all as clobber_execute
from smartsim._core._cli.dbcli import execute as dbcli_execute
from smartsim._core._cli.info import execute as info_execute
from smartsim._core._cli.plugin import plugins
from smartsim._core._cli.site import execute as site_execute
from smartsim._core._cli.utils import MenuItemConfig
from smartsim._core._cli.validate import configure_parser as validate_parser
from smartsim._core._cli.validate import execute as validate_execute


class SmartCli:
    def __init__(self, menu: t.List[MenuItemConfig]) -> None:
        self.menu: t.Dict[str, MenuItemConfig] = {}
        self.parser = argparse.ArgumentParser(
            prog="smart",
            description="SmartSim command line interface",
        )

        self.subparsers = self.parser.add_subparsers(
            dest="command",
            required=True,
            metavar="<command>",
            help="Available commands",
        )

        self.register_menu_items(menu)
        self.register_menu_items([plugin() for plugin in plugins])

    def execute(self, cli_args: t.List[str]) -> int:
        if len(cli_args) < 2:
            self.parser.print_help()
            return os.EX_USAGE

        app_args = cli_args[1:]  # exclude the path to executable
        subcommand = cli_args[1]  # first positional arg is the subcommand

        menu_item = self.menu.get(subcommand, None)
        if not menu_item:
            self.parser.print_help()
            return os.EX_USAGE

        args = argparse.Namespace()
        unparsed_args = []

        if menu_item.is_plugin:
            unparsed_args = app_args[1:]
        else:
            args = self.parser.parse_args(app_args)

        return menu_item.handler(args, unparsed_args)

    def _register_menu_item(self, item: MenuItemConfig) -> None:
        parser = self.subparsers.add_parser(
            item.command, description=item.description, help=item.description
        )
        if item.configurator:
            item.configurator(parser)

        if item.command in self.menu:
            raise ValueError(f"{item.command} cannot overwrite existing CLI command")

        self.menu[item.command] = item

    def register_menu_items(self, menu_items: t.List[MenuItemConfig]) -> None:
        for item in menu_items:
            self._register_menu_item(item)


def default_cli() -> SmartCli:
    menu = [
        MenuItemConfig(
            "build",
            "Build SmartSim dependencies (Redis, RedisAI, ML runtimes)",
            build_execute,
            build_parser,
        ),
        MenuItemConfig(
            "clean",
            "Remove previous ML runtime installation",
            clean_execute,
            clean_parser,
        ),
        MenuItemConfig(
            "dbcli",
            "Print the path to the redis-cli binary",
            dbcli_execute,
        ),
        MenuItemConfig(
            "site",
            "Print the installation site of SmartSim",
            site_execute,
        ),
        MenuItemConfig(
            "clobber",
            "Remove all previous dependency installations",
            clobber_execute,
        ),
        MenuItemConfig(
            "validate",
            "Run a simple SmartSim experiment to confirm that it is built correctly",
            validate_execute,
            validate_parser,
        ),
        MenuItemConfig(
            "info",
            "Display information about the current SmartSim installation",
            info_execute,
        ),
    ]

    return SmartCli(menu)
