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

# Generation Strategies
import random
import typing as t

from itertools import product


# create permutations of all parameters
# single model if parameters only have one value
def create_all_permutations(
    param_names: t.List[str], param_values: t.List[t.List[str]], _n_models: int = 0
) -> t.List[t.Dict[str, str]]:
    perms = list(product(*param_values))
    all_permutations = []
    for permutation in perms:
        temp_model = dict(zip(param_names, permutation))
        all_permutations.append(temp_model)
    return all_permutations


def step_values(
    param_names: t.List[str], param_values: t.List[t.List[str]], _n_models: int = 0
) -> t.List[t.Dict[str, str]]:
    permutations = []
    for param_value in zip(*param_values):
        permutations.append(dict(zip(param_names, param_value)))
    return permutations


def random_permutations(
    param_names: t.List[str], param_values: t.List[t.List[str]], n_models: int = 0
) -> t.List[t.Dict[str, str]]:
    permutations = create_all_permutations(param_names, param_values)

    # sample from available permutations if n_models is specified
    if n_models and n_models < len(permutations):
        permutations = random.sample(permutations, n_models)

    return permutations
