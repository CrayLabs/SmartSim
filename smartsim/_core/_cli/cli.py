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
import sys
import typing as t

# from pkg_resources import require

# import smartsim._core._cli as cli
from smartsim._core._cli.utils import MenuItemConfig
from smartsim._core._cli.clean import (
    configure_parser as clean_parser,
    execute as clean_execute,
    execute_all as clobber_execute,
)
import smartsim._core._cli.clean as clean
from smartsim._core._cli.site import execute as site_execute
from smartsim._core._cli.dbcli import execute as dbcli_execute
from smartsim._core._cli.build import configure_parser as build_parser, execute as build_execute

class SmartCli:
    def __init__(self, menu: t.List[MenuItemConfig]) -> None:
        self.menu: t.Dict[str, MenuItemConfig] = {item.command: item for item in menu}
        parser = argparse.ArgumentParser(
            prog="smart",
            description="SmartSim command line interface",
        )
        self.parser = parser

        subparsers = parser.add_subparsers(dest="command",
                                           required=True,
                                           metavar="<command>",
                                           help="Available commands")

        for cmd, item in self.menu.items():
            p = subparsers.add_parser(cmd, description=item.help, help=item.help)
            if item.configurator:
                item.configurator(p)

    def execute(self, cli_args: t.List[str]) -> int:
        if len(cli_args) < 2:
            self.parser.print_help()
            return 0

        app_args = cli_args[1:]
        args = self.parser.parse_args(app_args)

        if not (menu_item := self.menu.get(app_args[0], None)):
            self.parser.print_help()
            return 0

        return menu_item.handler(args)


def default_cli() -> SmartCli:
    build = MenuItemConfig("build",
                           "Build SmartSim dependencies (Redis, RedisAI, ML runtimes)",
                           build_execute,
                           build_parser)
    
    cleanx = MenuItemConfig("clean",
                            "Remove previous ML runtime installation",
                            clean_execute,
                            clean_parser)
    
    dbcli = MenuItemConfig("dbcli",
                           "Print the path to the redis-cli binary",
                           dbcli_execute)
    
    site = MenuItemConfig("site",
                          "Print the installation site of SmartSim",
                          site_execute)
    
    clobber = MenuItemConfig("clobber",
                             "Remove all previous dependency installations",
                             clobber_execute)
    
    menu = [build, cleanx, dbcli, site, clobber]
    smart_cli = SmartCli(menu)
    return smart_cli
