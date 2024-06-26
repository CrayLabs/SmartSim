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

import base64
import os
import pickle
import typing as t

from dragon.fli import FLInterface # pylint: disable=all

from smartsim._core.mli.infrastructure.storage.featurestore import FeatureStore


class EnvironmentConfigLoader:
    """
    Facilitates the loading of a FeatureStore and Queue
    into the WorkerManager.
    """

    def __init__(self) -> None:
        self._feature_store_descriptor = os.getenv("SSFeatureStore", None)
        self._queue_descriptor = os.getenv("SSQueue", None)
        self.feature_store: t.Optional[FeatureStore] = None
        self.queue: t.Optional["FLInterface"] = None

    def get_feature_store(self) -> t.Optional[FeatureStore]:
        """Loads the Feature Store previously set in SSFeatureStore"""
        if self._feature_store_descriptor is not None:
            self.feature_store = pickle.loads(
                base64.b64decode(self._feature_store_descriptor)
            )
        return self.feature_store

    def get_queue(self) -> t.Optional["FLInterface"]:
        """Returns the Queue  previously set in SSQueue"""
        if self._queue_descriptor is not None:
            return FLInterface.attach(base64.b64decode(self._queue_descriptor))
        return self.queue
