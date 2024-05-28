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


from os import getcwd, name

import pytest

from smartsim import Experiment
from smartsim.entity import EntityList
from smartsim.settings import RunSettings

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


def test_entity_list_init():
    with pytest.raises(NotImplementedError):
        ent_list = EntityList("list", getcwd(), perm_strat="all_perm")


def test_entity_list_getitem():
    """EntityList.__getitem__ is overridden in Ensemble, so we had to pass an instance of Ensemble
    to EntityList.__getitem__ in order to add test coverage to EntityList.__getitem__
    """
    exp = Experiment("name")
    ens_settings = RunSettings("python")
    ensemble = exp.create_ensemble("name", replicas=4, run_settings=ens_settings)
    assert ensemble.__getitem__("name_3") == ensemble["name_3"]
