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

from pkg_resources import require

import smartsim._core._cli as cli
from smartsim._core._cli.dbcli import DbCLI
from smartsim._core._cli.utils import get_install_path, MenuItem


class SmartCli:
    def __init__(self, menu: t.List[t.Type[MenuItem]]) -> None:
        self.menu = {item.command(): item for item in menu}
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
            # usage = "smart <command> [<args>]"
            p = subparsers.add_parser(cmd, 
                                      description=item.desc(), 
                                      help=item.help(),
            )
            item.configure_parser(p)

    def execute(self) -> None:
        if len(sys.argv) < 2:
            self.parser.print_help()
            sys.exit(0)

        app_args = sys.argv[1:]        
        args = self.parser.parse_args(app_args)

        if not (handler := self.menu.get(app_args[0], None)):
            self.parser.print_help()
            sys.exit(0)

        handler().execute(args)
        sys.exit(0)


def main() -> None:
    menu: t.Type[MenuItem] = [cli.Build,
                              cli.Clean,
                              cli.DbCLI,
                              cli.Site,
                              cli.Clobber]
    smart_cli = SmartCli(menu)
    smart_cli.execute()
