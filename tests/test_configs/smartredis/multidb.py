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

# import io
import os

# from smartsim.error.errors import
from smartredis import ConfigOptions, Client


# # taken from https://pytorch.org/docs/master/generated/torch.jit.trace.html
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv = nn.Conv2d(1, 1, 3)

#     def forward(self, x):
#         return self.conv(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SmartRedis ensemble producer process."
    )
    parser.add_argument("--exchange", action="store_true")
    args = parser.parse_args()

    opts1 = ConfigOptions.create_from_environment("testdb_reg")
    opts2 = ConfigOptions.create_from_environment("testdb_colo")

    c1 = Client(opts1, logger_name="SmartSim")
    c2 = Client(opts2, logger_name="SmartSim")


# jpTODO: add warning/error?
# if "SSKEYIN_testdb_reg" not in os.environ:
#     print("no env var found")
#     # raise EnvError(
#     #     "No env var found"
#     # )

# if "SSKEYOUT_testdb_reg" not in os.environ:
#     print("no env var found")
#     # raise EnvError(
#     #     "No env var found"
#     # )

# if "SSDB_testdb_reg" not in os.environ:
#     print("no env var found")
#     # raise EnvError(
#     #     "No env var found"
#     # )

# if "SR_DB_TYPE_testdb_reg" not in os.environ:
#     print("no env var found")
#     # raise EnvError(
#     #     "No env var found"
#     # )
