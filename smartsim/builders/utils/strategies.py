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
from dataclasses import dataclass, field

from smartsim.error import errors


@dataclass(frozen=True)
class ParamSet:
    """
    Represents a set of file parameters and execution arguments as parameters.
    """

    params: dict[str, str] = field(default_factory=dict)
    exe_args: dict[str, list[str]] = field(default_factory=dict)


# Type alias for the shape of a permutation strategy callable
PermutationStrategyType = t.Callable[
    [t.Mapping[str, t.Sequence[str]], t.Mapping[str, t.Sequence[t.Sequence[str]]], int],
    list[ParamSet],
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
        params: t.Mapping[str, t.Sequence[str]],
        exe_args: t.Mapping[str, t.Sequence[t.Sequence[str]]],
        n_permutations: int = -1,
    ) -> list[ParamSet]:
        try:
            permutations = fn(params, exe_args, n_permutations)
        except Exception as e:
            raise errors.UserStrategyError(str(fn)) from e
        if not isinstance(permutations, list) or not all(
            isinstance(permutation, ParamSet) for permutation in permutations
        ):
            raise errors.UserStrategyError(str(fn))
        return permutations

    return _impl


@_register("all_perm")
def create_all_permutations(
    params: t.Mapping[str, t.Sequence[str]],
    exe_arg: t.Mapping[str, t.Sequence[t.Sequence[str]]],
    n_permutations: int = -1,
) -> list[ParamSet]:
    """Take two mapping parameters to possible values and return a sequence of
    all possible permutations of those parameters.
    For example calling:
    .. highlight:: python
    .. code-block:: python
        create_all_permutations({"SPAM": ["a", "b"],
                                 "EGGS": ["c", "d"]},
                                 {"EXE": [["a"], ["b", "c"]],
                                 "ARGS": [["d"], ["e", "f"]]},
                                 1
                                 )
    Would result in the following permutations (not necessarily in this order):
    .. highlight:: python
    .. code-block:: python
        [ParamSet(params={'SPAM': 'a', 'EGGS': 'c'},
         exe_args={'EXE': ['a'], 'ARGS': ['d']})]
    :param file_params: A mapping of file parameter names to possible values
    :param exe_arg_params: A mapping of exe arg parameter names to possible values
    :param n_permutations: The maximum number of permutations to sample from
        the sequence of all permutations
    :return: A sequence of ParamSets of all possible permutations
    """
    file_params_permutations = itertools.product(*params.values())
    param_zip = (
        dict(zip(params, permutation)) for permutation in file_params_permutations
    )

    exe_arg_params_permutations = itertools.product(*exe_arg.values())
    exe_arg_params_permutations_ = (
        tuple(map(list, sequence)) for sequence in exe_arg_params_permutations
    )
    exe_arg_zip = (
        dict(zip(exe_arg, permutation)) for permutation in exe_arg_params_permutations_
    )

    combinations = itertools.product(param_zip, exe_arg_zip)
    param_set: t.Iterable[ParamSet] = (
        ParamSet(file_param, exe_arg) for file_param, exe_arg in combinations
    )
    if n_permutations >= 0:
        param_set = itertools.islice(param_set, n_permutations)
    return list(param_set)


@_register("step")
def step_values(
    params: t.Mapping[str, t.Sequence[str]],
    exe_args: t.Mapping[str, t.Sequence[t.Sequence[str]]],
    n_permutations: int = -1,
) -> list[ParamSet]:
    """Take two mapping parameters to possible values and return a sequence of
    stepped values until a possible values sequence runs out of possible
    values.
    For example calling:
    .. highlight:: python
    .. code-block:: python
        step_values({"SPAM": ["a", "b"],
                     "EGGS": ["c", "d"]},
                     {"EXE": [["a"], ["b", "c"]],
                     "ARGS": [["d"], ["e", "f"]]},
                     1
                     )
    Would result in the following permutations:
    .. highlight:: python
    .. code-block:: python
        [ParamSet(params={'SPAM': 'a', 'EGGS': 'c'},
                  exe_args={'EXE': ['a'], 'ARGS': ['d']})]
    :param file_params: A mapping of file parameter names to possible values
    :param exe_arg_params: A mapping of exe arg parameter names to possible values
    :param n_permutations: The maximum number of permutations to sample from
        the sequence of step permutations
    :return: A sequence of ParamSets of stepped values
    """
    param_zip: t.Iterable[tuple[str, ...]] = zip(*params.values())
    param_zip_ = (dict(zip(params, step)) for step in param_zip)

    exe_arg_zip: t.Iterable[tuple[t.Sequence[str], ...]] = zip(*exe_args.values())
    exe_arg_zip_ = (map(list, sequence) for sequence in exe_arg_zip)
    exe_arg_zip__ = (dict(zip(exe_args, step)) for step in exe_arg_zip_)

    param_set: t.Iterable[ParamSet] = (
        ParamSet(file_param, exe_arg)
        for file_param, exe_arg in zip(param_zip_, exe_arg_zip__)
    )
    if n_permutations >= 0:
        param_set = itertools.islice(param_set, n_permutations)
    return list(param_set)


@_register("random")
def random_permutations(
    params: t.Mapping[str, t.Sequence[str]],
    exe_args: t.Mapping[str, t.Sequence[t.Sequence[str]]],
    n_permutations: int = -1,
) -> list[ParamSet]:
    """Take two mapping parameters to possible values and return a sequence of
    length `n_permutations`  sampled randomly from all possible permutations
    :param file_params: A mapping of file parameter names to possible values
    :param exe_arg_params: A mapping of exe arg parameter names to possible values
    :param n_permutations: The maximum number of permutations to sample from
        the sequence of all permutations
    :return: A sequence of ParamSets of sampled permutations
    """
    permutations = create_all_permutations(params, exe_args, -1)
    if 0 <= n_permutations < len(permutations):
        permutations = random.sample(permutations, n_permutations)
    return permutations
