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

import functools
import itertools
import random
import typing as t

from smartsim.error import errors

# Type alias for the shape of a permutation strategy callable
PermutationStrategyType: t.TypeAlias = t.Callable[
    [t.Mapping[str, t.Sequence[str]], int], list[dict[str, str]]
]

# Map of globally registered strategy names to registered strategy callables
_REGISTERED_STRATEGIES: t.Final[dict[str, PermutationStrategyType]] = {}


def _register(name: str) -> t.Callable[
    [PermutationStrategyType],
    PermutationStrategyType,
]:
    """Create a decorator to globally register a permutation strategy under a
    given name.

    :param name: The name under which to register a strategy
    :return: A decorator to register a permutation strategy function
    """

    def _impl(fn: PermutationStrategyType) -> PermutationStrategyType:
        """Add a strategy function to the globally registered strategies under
        the `name` caught in the closure.

        :param fn: A permutation strategy
        :returns: The original strategy, unaltered
        :raises ValueError: A strategy under name caught in the closure has
            already been registered
        """
        if name in _REGISTERED_STRATEGIES:
            msg = f"A strategy with the name '{name}' has already been registered"
            raise ValueError(msg)
        _REGISTERED_STRATEGIES[name] = fn
        return fn

    return _impl


def resolve(strategy: str | PermutationStrategyType) -> PermutationStrategyType:
    """Look-up or sanitize a permutation strategy:

        - When `strategy` is a `str` it will look for a globally registered
          strategy function by that name.

        - When `strategy` is a `callable` it is will return a sanitized
          strategy function.

    :param strategy: The name of a registered strategy or a custom
        permutation strategy
    :return: A valid permutation strategy callable
    """
    if callable(strategy):
        return _make_sanitized_custom_strategy(strategy)
    try:
        return _REGISTERED_STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"Failed to find an ensembling strategy by the name of '{strategy}'."
            f"All known strategies are:\n{', '.join(_REGISTERED_STRATEGIES)}"
        ) from None


def _make_sanitized_custom_strategy(
    fn: PermutationStrategyType,
) -> PermutationStrategyType:
    """Take a callable that satisfies the shape of a permutation strategy and
    return a sanitized version for future callers.

    The sanitized version of the permutation strategy will intercept any
    exceptions raised by the original permutation and re-raise a
    `UserStrategyError`.

    The sanitized version will also check the type of the value returned from
    the original callable, and if it does conform to the expected return type,
    a `UserStrategyError` will be raised.

    :param fn: A custom user strategy function
    :return: A sanitized version of the custom strategy function
    """

    @functools.wraps(fn)
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
    """Take a mapping parameters to possible values and return a sequence of
    all possible permutations of those parameters.

    For example calling:

    .. highlight:: python
    .. code-block:: python

        create_all_permutations({"A": ["1", "2"],
                                 "B": ["3", "4"]})

    Would result in the following permutations (not necessarily in this order):

    .. highlight:: python
    .. code-block:: python

        [{"A": "1", "B": "3"},
         {"A": "1", "B": "4"},
         {"A": "2", "B": "3"},
         {"A": "2", "B": "4"}]

    :param params: A mapping of parameter names to possible values
    :param _n_permutations: <ignored>
    :return: A sequence of mappings of all possible permutations
    """
    permutations = itertools.product(*params.values())
    return [dict(zip(params, permutation)) for permutation in permutations]


@_register("step")
def step_values(
    params: t.Mapping[str, t.Sequence[str]], _n_permutations: int = 0
) -> list[dict[str, str]]:
    """Take a mapping parameters to possible values and return a sequence of
    stepped values until a possible values sequence runs out of possible
    values.

    For example calling:

    .. highlight:: python
    .. code-block:: python

        step_values({"A": ["1", "2"],
                     "B": ["3", "4"]})

    Would result in the following permutations:

    .. highlight:: python
    .. code-block:: python

        [{"A": "1", "B": "3"},
         {"A": "2", "B": "4"}]

    :param params: A mapping of parameter names to possible values
    :param _n_permutations: <ignored>
    :return: A sequence of mappings of stepped values
    """
    steps = zip(*params.values())
    return [dict(zip(params, step)) for step in steps]


@_register("random")
def random_permutations(
    params: t.Mapping[str, t.Sequence[str]], n_permutations: int = 0
) -> list[dict[str, str]]:
    """Take a mapping parameters to possible values and return a sequence of
    length `n_permutations`  sampled randomly from all possible permutations

    :param params: A mapping of parameter names to possible values
    :param n_permutations: The maximum number of permutations to sample from
        the sequence of all permutations
    :return: A sequence of mappings of sampled permutations
    """
    permutations = create_all_permutations(params, 0)

    # sample from available permutations if n_permutations is specified
    if 0 < n_permutations < len(permutations):
        permutations = random.sample(permutations, n_permutations)

    return permutations
