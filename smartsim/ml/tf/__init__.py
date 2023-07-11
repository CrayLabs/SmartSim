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

from ..._core._install.buildenv import Version_, Versioner
from ...error import SmartSimError
from ...log import get_logger

logger = get_logger(__name__)

vers = Versioner()
TF_VERSION = vers.TENSORFLOW

try:
    import tensorflow as tf

    installed_tf = Version_(tf.__version__)
    assert installed_tf >= "2.4.0"

except ImportError:  # pragma: no cover
    raise ModuleNotFoundError(
        f"TensorFlow {TF_VERSION} is not installed. "
        "Please install it to use smartsim.tf"
    ) from None
except AssertionError:  # pragma: no cover
    msg = (
        f"TensorFlow >= {TF_VERSION} is required for smartsim. "
        f"tf, you have {tf.__version__}"
    )
    raise SmartSimError() from None


# pylint: disable=wrong-import-position
from .data import DynamicDataGenerator, StaticDataGenerator
from .utils import freeze_model, serialize_model
