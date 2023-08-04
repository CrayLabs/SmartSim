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


from contextlib import contextmanager

import smartredis

import smartsim._core._cli.test
from smartsim._core.utils.helpers import installed_redisai_backends


def test_cli_miniexp_doesnt_error_out_with_dev_build(
    local_db,
    fileutils,
    monkeypatch,
):
    """Presumably devs running the test suite have built SS correctly.
    This test runs the "mini-exp" shipped users through the CLI
    to ensure that it does not accidentally report false positive/negatives
    """

    @contextmanager
    def _mock_make_managed_local_orc(*a, **kw):
        client_addr ,= local_db.get_address()
        yield smartredis.Client(address=client_addr, cluster=False)

    monkeypatch.setattr(
        smartsim._core._cli.test,
        "_make_managed_local_orc",
        _mock_make_managed_local_orc,
    )
    backends = installed_redisai_backends()
    db_port ,= local_db.ports

    smartsim._core._cli.test.test_install(
        # Shouldn't matter bc making the managed is stubbed
        # but best to give it "correct" vals for safety
        location=fileutils.get_test_dir(),
        port=db_port,
        # Test the backends the dev has installed
        with_tf="tensorflow" in backends,
        with_pt="torch" in backends,
        with_onnx="onnxruntime" in backends,
    )
