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

import os
import sys

from smartsim._core._cli.cli import default_cli
from smartsim._core._cli.utils import SMART_LOGGER_FORMAT
from smartsim.error.errors import SmartSimCLIActionCancelled
from smartsim.log import get_logger

logger = get_logger("Smart", fmt=SMART_LOGGER_FORMAT)


def main() -> int:
    smart_cli = default_cli()
    exception_trace_back_msg = "SmartSim exited with the following exception info:"

    try:
        return smart_cli.execute(sys.argv)
    except SmartSimCLIActionCancelled as ssi:
        logger.info(str(ssi))
        logger.debug(exception_trace_back_msg, exc_info=ssi)
    except KeyboardInterrupt as e:
        logger.info("SmartSim was terminated by user")
        logger.debug(exception_trace_back_msg, exc_info=e)
    return os.EX_OK


if __name__ == "__main__":
    main()
