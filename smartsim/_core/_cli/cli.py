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

from smartsim._core._cli.build import Build
from smartsim._core._cli.clean import Clean
from smartsim._core._cli.utils import get_install_path


def _usage() -> str:
    usage = [
        "smart <command> [<args>]\n",
        "Commands:",
        "\tbuild       Build SmartSim dependencies (Redis, RedisAI, ML runtimes)",
        "\tclean       Remove previous ML runtime installation",
        "\tclobber     Remove all previous dependency installations",
        "\nDeveloper:",
        "\tsite        Print the installation site of SmartSim",
        "\tdbcli       Print the path to the redis-cli binary" "\n\n",
    ]
    return "\n".join(usage)


class SmartCli:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description="SmartSim command line interface", usage=_usage()
        )

        parser.add_argument("command", help="Subcommand to run")

        # smart
        if len(sys.argv) < 2:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            parser.print_help()
            sys.exit(0)
        getattr(self, args.command)()

    def build(self) -> None:
        Build()
        sys.exit(0)

    def clean(self) -> None:
        Clean()
        sys.exit(0)

    def clobber(self) -> None:
        Clean(clean_all=True)
        sys.exit(0)

    def site(self) -> None:
        print(get_install_path())
        sys.exit(0)

    def dbcli(self) -> None:
        bin_path = get_install_path() / "_core" / "bin"
        for option in bin_path.iterdir():
            if option.name in ("redis-cli", "keydb-cli"):
                print(option)
                sys.exit(0)
        print("Database (Redis or KeyDB) dependencies not found")
        sys.exit(1)


def main() -> None:
    SmartCli()
