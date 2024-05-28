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
import os
import subprocess
import typing as t

from smartsim._core.config import CONFIG


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Builds the parser for the command"""
    parser.add_argument(
        "--dragon",
        action="store_true",
        default=False,
        help="Terminate Dragon environment resources if"
        "any remain after experiment completion",
    )


def _do_dragon_teardown() -> int:
    """Run dragon-cleanup script to destroy all remaining dragon resources"""
    env = os.environ.copy()
    dragon_cleanup = next(CONFIG.core_path.rglob("dragon-cleanup"), None)
    if dragon_cleanup is None:
        print("dragon-cleanup not found. Skipping cleanup")
        return 0

    # Use popen to avoid `dragon-cleanup` doing a kill on all
    # python processes, terminating `smart teardown`, and the
    # subprocess handling `dragon-cleanup`. Child processes using
    # subprocess.run are killed and cleanup is interrupted
    with subprocess.Popen(
        [str(dragon_cleanup.absolute())],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        process.wait()
        return process.returncode


def execute(
    args: argparse.Namespace, _unparsed_args: t.Optional[t.List[str]] = None, /
) -> int:
    if args.dragon:
        return _do_dragon_teardown()

    return 0
