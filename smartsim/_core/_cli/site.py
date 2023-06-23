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

from smartsim._core._cli.utils import get_install_path, MenuItem
from smartsim.log import get_logger

smart_logger_format = "[%(name)s] %(levelname)s %(message)s"
logger = get_logger("Smart", fmt=smart_logger_format)


class Site(MenuItem):
    def execute(self, args: argparse.Namespace) -> None:
        print(get_install_path())

    @staticmethod
    def command() -> str:
        return "site"
    
    @staticmethod
    def help() -> str:
        return "(help for) Print the installation site of SmartSim"
    
    @staticmethod
    def usage() -> str:
        return f"smart {Site.command()} [options]"
    
    @staticmethod
    def desc() -> str:
        return "Print the installation site of SmartSim"

    @staticmethod
    def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Builds the parser for the command"""
        return parser
