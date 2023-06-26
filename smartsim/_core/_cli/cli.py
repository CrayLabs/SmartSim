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
from smartsim._core._cli.utils import MenuItemConfig, clean, get_install_path, get_db_path
from smartsim._core._cli.clean import configure_parser as clean_parser

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
            item.configurator(p)

    def execute(self) -> None:
        if len(sys.argv) < 2:
            self.parser.print_help()
            sys.exit(0)

        app_args = sys.argv[1:]        
        args = self.parser.parse_args(app_args)

        if not (menu_item := self.menu.get(app_args[0], None)):
            self.parser.print_help()
            sys.exit(0)

        menu_item.handler(args)
        sys.exit(0)


def main() -> None:
    build = MenuItemConfig("build",
                           "Build SmartSim dependencies (Redis, RedisAI, ML runtimes)",
                           lambda args: print('fake build'),
                           lambda p: p)
    
    cleanx = MenuItemConfig("clean",
                            "Remove previous ML runtime installation",
                            lambda args: clean(get_install_path() / "_core", _all=args.clobber),
                            clean_parser)
    
    dbcli = MenuItemConfig("dbcli",
                           "Print the path to the redis-cli binary",
                           lambda args: print(get_db_path()),
                           lambda p: p)
    
    site = MenuItemConfig("site",
                          "Print the installation site of SmartSim",
                          lambda args: print(get_install_path()),
                          lambda p: p)
    
    clobber = MenuItemConfig("clobber",
                             "Remove all previous dependency installations",
                             lambda args: clean(get_install_path() / "_core", _all=True),
                             lambda p: p)
    
    menu = [build, cleanx, dbcli, site, clobber]
    smart_cli = SmartCli(menu)
    smart_cli.execute()
