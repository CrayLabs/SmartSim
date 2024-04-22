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

import io
import json
import random
import string
import textwrap

import pytest

from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.entity.dbnode import DBNode, LaunchedShardData
from smartsim.error.errors import SmartSimError

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


def test_parse_db_host_error():
    orc = Orchestrator()
    orc.entities[0].path = "not/a/path"
    # Fail to obtain database hostname
    with pytest.raises(SmartSimError):
        orc.entities[0].host


def test_hosts(test_dir, wlmutils):
    exp_name = "test_hosts"
    exp = Experiment(exp_name, exp_path=test_dir)

    orc = Orchestrator(port=wlmutils.get_test_port(), interface="lo", launcher="local")
    orc.set_path(test_dir)
    exp.start(orc)

    hosts = []
    try:
        hosts = orc.hosts
        assert len(hosts) == orc.db_nodes == 1
    finally:
        # stop the database even if there is an error raised
        exp.stop(orc)
        orc.remove_stale_files()


def _random_shard_info():
    rand_string = lambda: "".join(random.choices(string.ascii_letters, k=10))
    rand_num = lambda: random.randint(1000, 9999)
    flip_coin = lambda: random.choice((True, False))

    return LaunchedShardData(
        name=rand_string(),
        hostname=rand_string(),
        port=rand_num(),
        cluster=flip_coin(),
    )


def test_launched_shard_info_can_be_serialized():
    shard_data = _random_shard_info()
    shard_data_from_str = LaunchedShardData(
        **json.loads(json.dumps(shard_data.to_dict()))
    )

    assert shard_data is not shard_data_from_str
    assert shard_data == shard_data_from_str


@pytest.mark.parametrize("limit", [None, 1])
def test_db_node_can_parse_launched_shard_info(limit):
    rand_shards = [_random_shard_info() for _ in range(3)]
    with io.StringIO(textwrap.dedent("""\
            This is some file like str
            --------------------------

            SMARTSIM_ORC_SHARD_INFO: {}
            ^^^^^^^^^^^^^^^^^^^^^^^
            We should be able to parse the serialized
            launched db info from this file if the line is
            prefixed with this tag.

            Here are two more for good measure:
            SMARTSIM_ORC_SHARD_INFO: {}
            SMARTSIM_ORC_SHARD_INFO: {}

            All other lines should be ignored.
            """).format(*(json.dumps(s.to_dict()) for s in rand_shards))) as stream:
        parsed_shards = DBNode._parse_launched_shard_info_from_iterable(stream, limit)
    if limit is not None:
        rand_shards = rand_shards[:limit]
    assert rand_shards == parsed_shards


def test_set_host():
    orc = Orchestrator()
    orc.entities[0].set_hosts(["host"])
    assert orc.entities[0].host == "host"


@pytest.mark.parametrize("nodes, mpmd", [[3, False], [3, True], [1, False]])
def test_db_id_and_name(mpmd, nodes, wlmutils):
    if nodes > 1 and wlmutils.get_test_launcher() not in pytest.wlm_options:
        pytest.skip(reason="Clustered DB can only be checked on WLMs")
    orc = Orchestrator(
        db_identifier="test_db",
        db_nodes=nodes,
        single_cmd=mpmd,
        launcher=wlmutils.get_test_launcher(),
    )
    for i, node in enumerate(orc.entities):
        assert node.name == f"{orc.name}_{i}"
        assert node.db_identifier == orc.db_identifier
