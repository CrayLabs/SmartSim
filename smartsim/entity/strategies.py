# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

from itertools import product


# create permutations of all parameters
# single model if parameters only have one value
def create_all_permutations(param_names, param_values):
    perms = list(product(*param_values))
    all_permutations = []
    for p in perms:
        temp_model = dict(zip(param_names, p))
        all_permutations.append(temp_model)
    return all_permutations


def step_values(param_names, param_values):
    permutations = []
    for p in zip(*param_values):
        permutations.append(dict(zip(param_names, p)))
    return permutations


def random_permutations(param_names, param_values, n_models):
    import random

    # first, check if we've requested more values than possible.
    perms = list(product(*param_values))
    if n_models >= len(perms):
        return create_all_permutations(param_names, param_values)
    else:
        permutations = []
        permutation_strings = set()
        while len(permutations) < n_models:
            model_dict = dict(
                zip(
                    param_names,
                    map(lambda x: x[random.randint(0, len(x) - 1)], param_values),
                )
            )
            if str(model_dict) not in permutation_strings:
                permutation_strings.add(str(model_dict))
                permutations.append(model_dict)
        return permutations
