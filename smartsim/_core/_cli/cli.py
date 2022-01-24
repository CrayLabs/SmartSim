#!/usr/bin/env python

import argparse
import sys

from smartsim._core._cli.build import Build
from smartsim._core._cli.clean import Clean
from smartsim._core._cli.utils import get_install_path


def _usage():
    usage = [
        "smart <command> [<args>]\n",
        "Commands:",
        "\tbuild     Build SmartSim dependencies (Redis, RedisAI, ML runtimes)",
        "\tclean     Remove previous ML runtime installation",
        "\tclobber   Remove all previous dependency installations",
        "\nDeveloper:",
        "\tsite      Print the installation site of SmartSim",
        "\tdbcli     Print the path to the redis-cli binary" "\n\n",
    ]
    return "\n".join(usage)


class SmartCli:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="SmartSim command line interface", usage=_usage()
        )

        parser.add_argument("command", help="Subcommand to run")

        # smart
        if len(sys.argv) < 2:
            parser.print_help()
            exit(0)

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            parser.print_help()
            exit(0)
        getattr(self, args.command)()

    def build(self):
        Build()
        exit(0)

    def clean(self):
        Clean()
        exit(0)

    def clobber(self):
        Clean(clean_all=True)
        exit(0)

    def site(self):
        print(get_install_path())
        exit(0)

    def dbcli(self):
        install_path = get_install_path()
        script_path = install_path.joinpath("_core/bin/redis-cli").resolve()
        if script_path.is_file():
            print(script_path)
            exit(0)
        else:
            print("Redis dependencies not found")
            exit(1)


def main():
    SmartCli()
