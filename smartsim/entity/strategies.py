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

# Generation Strategies

from __future__ import annotations

import itertools
import random
import typing as t

from smartsim.error import errors

TPermutationStrategy = t.Callable[
    [t.Mapping[str, t.Sequence[str]], int], list[dict[str, str]]
]

_REGISTERED_STRATEGIES: t.Final[dict[str, TPermutationStrategy]] = {}


def _register(name: str) -> t.Callable[
    [TPermutationStrategy],
    TPermutationStrategy,
]:
    def _impl(fn: TPermutationStrategy) -> TPermutationStrategy:
        if name in _REGISTERED_STRATEGIES:
            raise ValueError(
                f"A strategy with the name '{name}' has already been registered"
            )
        _REGISTERED_STRATEGIES[name] = fn
        return fn

    return _impl


def resolve(strategy: str | TPermutationStrategy) -> TPermutationStrategy:
    if callable(strategy):
        return _make_safe_custom_strategy(strategy)
    try:
        return _REGISTERED_STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"Failed to find an ensembling strategy by the name of '{strategy}'."
            f"All known strategies are:\n{', '.join(_REGISTERED_STRATEGIES)}"
        ) from None


def _make_safe_custom_strategy(fn: TPermutationStrategy) -> TPermutationStrategy:
    def _impl(
        params: t.Mapping[str, t.Sequence[str]], n_permutations: int
    ) -> list[dict[str, str]]:
        try:
            permutations = fn(params, n_permutations)
        except Exception as e:
            raise errors.UserStrategyError(str(fn)) from e
        if not isinstance(permutations, list) or not all(
            isinstance(permutation, dict) for permutation in permutations
        ):
            raise errors.UserStrategyError(str(fn))
        return permutations

    return _impl


# create permutations of all parameters
# single application if parameters only have one value
@_register("all_perm")
def create_all_permutations(
    params: t.Mapping[str, t.Sequence[str]],
    _n_permutations: int = 0,
    # ^^^^^^^^^^^^^
    # TODO: Really don't like that this attr is ignored, but going to leave it
    #       as the original impl for now. Will change if requested!
) -> list[dict[str, str]]:
    permutations = itertools.product(*params.values())
    return [dict(zip(params, permutation)) for permutation in permutations]


@_register("step")
def step_values(
    params: t.Mapping[str, t.Sequence[str]], _n_permutations: int = 0
) -> list[dict[str, str]]:
    steps = zip(*params.values())
    return [dict(zip(params, step)) for step in steps]


@_register("random")
def random_permutations(
    params: t.Mapping[str, t.Sequence[str]], n_permutations: int = 0
) -> list[dict[str, str]]:
    permutations = create_all_permutations(params, 0)

    # sample from available permutations if n_permutations is specified
    if 0 < n_permutations < len(permutations):
        permutations = random.sample(permutations, n_permutations)

    return permutations
