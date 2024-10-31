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

from __future__ import annotations

import dataclasses
import typing as t

from smartsim._core.utils import helpers as _helpers

if t.TYPE_CHECKING:
    from smartsim._core.utils.launcher import LauncherProtocol
    from smartsim.types import LaunchedJobID


@dataclasses.dataclass(frozen=True)
class LaunchHistory:
    """A cache to manage and quickly look up which launched job ids were
    issued by which launcher
    """

    _id_to_issuer: dict[LaunchedJobID, LauncherProtocol[t.Any]] = dataclasses.field(
        default_factory=dict
    )

    def save_launch(
        self, launcher: LauncherProtocol[t.Any], id_: LaunchedJobID
    ) -> None:
        """Save a launcher and a launch job id that it issued for later
        reference.

        :param launcher: A launcher that started a job and issued an id for
            that job
        :param id_: The id of the launched job started by the launcher
        :raises ValueError: An id of equal value has already been saved
        """
        if id_ in self._id_to_issuer:
            raise ValueError("An ID of that value has already been saved")
        self._id_to_issuer[id_] = launcher

    def iter_past_launchers(self) -> t.Iterable[LauncherProtocol[t.Any]]:
        """Iterate over the unique launcher instances stored in history

        :returns: An iterator over unique launcher instances
        """
        return _helpers.unique(self._id_to_issuer.values())

    def group_by_launcher(
        self, ids: t.Collection[LaunchedJobID] | None = None, unknown_ok: bool = False
    ) -> dict[LauncherProtocol[t.Any], set[LaunchedJobID]]:
        """Return a mapping of launchers to launched job ids issued by that
        launcher.

        :param ids: The subset launch ids to group by common launchers.
        :param unknown_ok: If set to `True` and the history is unable to
            determine which launcher instance issued a requested launched job
            id, the history will silently omit the id from the returned
            mapping. If set to `False` a `ValueError` will be raised instead.
            Set to `False` by default.
        :raises ValueError: An unknown launch id was requested to be grouped by
            launcher, and `unknown_ok` is set to `False`.
        :returns: A mapping of launchers to collections of launched job ids
            that were issued by that launcher.
        """
        if ids is None:
            ids = self._id_to_issuer
        launchers_to_launched = _helpers.group_by(self._id_to_issuer.get, ids)
        unknown = launchers_to_launched.get(None, [])
        if unknown and not unknown_ok:
            formatted_unknown = ", ".join(unknown)
            msg = f"IDs {formatted_unknown} could not be mapped back to a launcher"
            raise ValueError(msg)
        return {k: set(v) for k, v in launchers_to_launched.items() if k is not None}
