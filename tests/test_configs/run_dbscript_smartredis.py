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

import numpy as np
from pytest import approx
from smartredis import Client


def main():
    # address should be set as we are launching through
    # SmartSim.
    client = Client(cluster=False)

    array = np.ones((1, 3, 3, 1)).astype(np.single)
    client.put_tensor("test_array", array)
    assert client.poll_model("test_script1", 500, 30)
    client.run_script("test_script1", "average", ["test_array"], ["test_output"])
    returned = client.get_tensor("test_output")
    assert returned == approx(np.mean(array))

    assert client.poll_model("test_script2", 500, 30)
    client.run_script("test_script2", "negate", ["test_array"], ["test_output"])
    returned = client.get_tensor("test_output")
    assert returned == approx(-array)

    if client.model_exists("test_func"):
        client.run_script("test_func", "timestwo", ["test_array"], ["test_output"])
        returned = client.get_tensor("test_output")
        assert returned == approx(2 * array)

    print(f"Test worked!")


if __name__ == "__main__":
    main()
