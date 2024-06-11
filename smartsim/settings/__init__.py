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

from .baseSettings import BaseSettings
from .batchSettings import BatchSettings
from .launchSettings import LaunchSettings

__all__ = ["LaunchSettings", "BaseSettings", "BatchSettings"]


# TODO Mock imports for compiling tests
class DragonRunSettings:
    pass


class QsubBatchSettings:
    pass


class SbatchSettings:
    pass


class Singularity:
    pass


class SettingsBase:
    pass


class AprunSettings:
    pass


class RunSettings:
    pass


class OrterunSettings:
    pass


class MpirunSettings:
    pass


class MpiexecSettings:
    pass


class JsrunSettings:
    pass


class BsubBatchSettings:
    pass


class PalsMpiexecSettings:
    pass


class SrunSettings:
    pass


class Container:
    pass


def create_batch_settings() -> None: ...
def create_run_settings() -> None: ...
