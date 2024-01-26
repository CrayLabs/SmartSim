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
import typing as t
import uuid

import pytest

from smartsim._core.entrypoints.telemetrymonitor import FileSink, JobEntity

# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.group_a


@pytest.fixture
def mock_entity(test_dir):
    def _mock_entity(
        host: str = "127.0.0.1", port: str = "6379", name: str = "", type: str = ""
    ):
        entity = JobEntity()
        entity.name = name if name else str(uuid.uuid4())
        entity.status_dir = test_dir
        entity.type = type
        entity.config = {
            "host": host,
            "port": port,
        }
        return entity

    return _mock_entity


@pytest.mark.asyncio
async def test_sink_null_entity() -> None:
    """Ensure the filesink handles a null entity as expected"""
    # entity = mock_entity(port=1234, name="e1")
    entity: t.Optional[JobEntity] = None

    with pytest.raises(ValueError) as ex:
        _ = FileSink(entity, "test.csv")

    assert "entity" in "".join(ex.value.args)


@pytest.mark.parametrize(
        "name,type",
        [
            pytest.param("e1", "", id="No entity type"),
            pytest.param("e2", "ensemble", id="Has entity type"),
        ]
)
@pytest.mark.asyncio
async def test_sink_null_filename(mock_entity, name, type) -> None:
    """Ensure the filesink handles a null filename as expected"""
    entity = mock_entity(port=1234, name=name, type=type)

    # pass null file path
    sink = FileSink(entity, None)
    await sink.save(key="value1")
    await sink.save(key="value2")
    await sink.save(key="value3")

    # show that logs are still written
    assert sink.path.exists()

    if type:
        assert type in str(sink.path)

    # show that the same file is written to
    content = sink.path.read_text()
    assert len(content.splitlines()) == 3


@pytest.mark.asyncio
async def test_sink_write(mock_entity) -> None:
    """Ensure the FileSink writes values to the output file as expected"""
    entity = mock_entity(port=1234, name="e1")

    sink = FileSink(entity, "test.csv")

    # all values are converted to strings before saving
    v1, v2, v3 = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    await sink.save(key1=v1, key2=v2, key3=v3)

    # show file was written
    path = sink.path
    assert path.exists()

    # show each value is found in the file
    content = path.read_text()
    for value in [v1, v2, v3]:
        assert str(value) in content


@pytest.mark.asyncio
async def test_sink_write_nonstring_input(mock_entity) -> None:
    """Ensure the FileSink writes values to the output file as expected
    when inputs are non-strings"""
    entity = mock_entity(port=1234, name="e1")

    sink = FileSink(entity, "test.csv")

    # v1, v2 are not converted to strings
    v1, v2 = 1, uuid.uuid4()
    await sink.save(key1=v1, key2=v2)

    # show file was written
    path = sink.path
    assert path.exists()

    # split down to individual elements to ensure expected default format
    content = path.read_text()
    lines = content.splitlines()
    line = lines[0].split(",")

    # show each value can be found
    for value in [v1, v2]:
        assert str(value) in line


@pytest.mark.asyncio
async def test_sink_write_no_inputs(mock_entity) -> None:
    """Ensure the FileSink writes to an output file without error if no
    values are supplied"""
    entity = mock_entity(port=1234, name="e1")
    sink = FileSink(entity, "test.csv")

    num_saves = 5
    for i in range(num_saves):
        await sink.save()

    path = sink.path
    assert path.exists()

    # show file was written
    content = path.read_text()

    # show a line was written for each call to save
    assert len(content.splitlines()) == num_saves


@pytest.mark.asyncio
async def test_sink_write_null_entity(mock_entity) -> None:
    """Ensure the FileSink writes to an output file without error if no
    values are supplied"""
    entity = mock_entity(port=1234, name="e1")
    sink = FileSink(entity, "test.csv")

    num_saves = 5
    for i in range(num_saves):
        await sink.save()

    path = sink.path
    assert path.exists()

    # show file was written
    content = path.read_text()

    # show a line was written for each call to save
    assert len(content.splitlines()) == num_saves