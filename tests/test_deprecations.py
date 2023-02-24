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


import pytest

from smartsim.database import (
    CobaltOrchestrator,
    LSFOrchestrator,
    PBSOrchestrator,
    SlurmOrchestrator,
)

tf_available = True
try:
    import tensorflow
except ImportError:
    tf_available = False


def test_deprecated_orchestrators(wlmutils):
    with pytest.deprecated_call():
        _ = SlurmOrchestrator(interface=wlmutils.get_test_interface())

    with pytest.deprecated_call():
        _ = LSFOrchestrator(interface=wlmutils.get_test_interface())

    with pytest.deprecated_call():
        _ = CobaltOrchestrator(interface=wlmutils.get_test_interface())

    with pytest.deprecated_call():
        _ = PBSOrchestrator(interface=wlmutils.get_test_interface())


@pytest.mark.skipif(not tf_available, reason="Requires TF to run")
def test_deprecated_tf():
    with pytest.deprecated_call():
        from smartsim.tf import freeze_model


def test_deprecated_constants():
    with pytest.deprecated_call():
        from smartsim import constants
