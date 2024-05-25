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

import pytest

from smartsim.database.orchestrator import Orchestrator
from smartsim.entity.entity import SmartSimEntity
from smartsim.entity.model import Model
from smartsim.error.errors import SSUnsupportedError
from smartsim.launchable import Job, Launchable
from smartsim.launchable.launchable import SmartSimObject
from smartsim.launchable.mpmdjob import MPMDJob
from smartsim.launchable.mpmdpair import MPMDPair
from smartsim.settings.base import RunSettings


def test_smartsimobject_init():
    ss_object = SmartSimObject()
    assert isinstance(ss_object, SmartSimObject)


def test_launchable_init():
    launchable = Launchable()
    assert isinstance(launchable, Launchable)


def test_job_init():
    entity = SmartSimEntity("test_name", None, None)
    launch_settings = RunSettings("echo", ["spam", "eggs"])
    job = Job(entity, launch_settings)
    assert isinstance(job, Job)


def test_add_mpmd_pair():
    """Test the creation of an MPMDJob and then adding an mpmd pair
    using add_mpmd_pair"""
    entity = SmartSimEntity("test_name", None, None)
    launch_settings = RunSettings("echo", ["spam", "eggs"])

    mpmd_job = MPMDJob()
    mpmd_job.add_mpmd_pair(entity, launch_settings)

    mpmd_pair = MPMDPair(entity, launch_settings)

    assert len(mpmd_job.mpmd_pairs) > 0

    assert mpmd_pair.entity == mpmd_job.mpmd_pairs[0].entity
    assert mpmd_pair.launch_settings == mpmd_job.mpmd_pairs[0].launch_settings


def test_mpmdpair_init():
    """Test the creation of an MPMDPair"""
    entity = SmartSimEntity("test_name", None, None)
    launch_settings = RunSettings("echo", ["spam", "eggs"])
    mpmd_pair = MPMDPair(entity, launch_settings)
    assert isinstance(mpmd_pair, MPMDPair)


def test_check_launcher():
    """Test that mpmd pairs that have the same launcher type can be added to an MPMD Job"""

    entity1 = SmartSimEntity("entity1", None, None)
    launch_settings1 = RunSettings("echo", ["hello", "world"], run_command="mpirun")
    entity2 = SmartSimEntity("entity2", None, None)
    launch_settings2 = RunSettings("echo", ["spam", "eggs"], run_command="mpirun")
    mpmd_pairs = []

    pair1 = MPMDPair(entity1, launch_settings1)
    mpmd_pairs.append(pair1)
    mpmd_job = MPMDJob(mpmd_pairs)
    # add a second mpmd pair to the mpmd job
    mpmd_job.add_mpmd_pair(entity2, launch_settings2)

    assert str(mpmd_job.mpmd_pairs[0].entity) == "entity1"
    assert str(mpmd_job.mpmd_pairs[1].entity) == "entity2"


def test_add_mpmd_pair_check_launcher_error():
    """Test that an error is raised when a pairs is added to an mpmd
    job using add_mpmd_pair that does not have the same launcher type"""
    mpmd_pairs = []
    entity1 = SmartSimEntity("entity1", None, None)
    launch_settings1 = RunSettings("echo", ["hello", "world"], run_command="srun")

    entity2 = SmartSimEntity("entity2", None, None)
    launch_settings2 = RunSettings("echo", ["spam", "eggs"], run_command="mpirun")

    pair1 = MPMDPair(entity1, launch_settings1)
    mpmd_pairs.append(pair1)
    mpmd_job = MPMDJob(mpmd_pairs)

    # Add a second mpmd pair to the mpmd job with a different launcher
    with pytest.raises(SSUnsupportedError):
        mpmd_job.add_mpmd_pair(entity2, launch_settings2)


def test_add_mpmd_pair_check_entity():
    """Test that mpmd pairs that have the same entity type can be added to an MPMD Job"""
    mpmd_pairs = []
    entity1 = Model("entity1", None, None)
    launch_settings1 = RunSettings("echo", ["hello", "world"], run_command="srun")

    entity2 = Model("entity2", None, None)
    launch_settings2 = RunSettings("echo", ["spam", "eggs"], run_command="srun")

    pair1 = MPMDPair(entity1, launch_settings1)
    mpmd_pairs.append(pair1)
    mpmd_job = MPMDJob(mpmd_pairs)

    # Add a second mpmd pair to the mpmd job
    mpmd_job.add_mpmd_pair(entity2, launch_settings2)

    assert isinstance(mpmd_job, MPMDJob)


def test_add_mpmd_pair_check_entity_error():
    """Test that an error is raised when a pairs is added to an mpmd job
    using add_mpmd_pair that does not have the same entity type"""
    mpmd_pairs = []
    entity1 = Model("entity1", None, None)
    launch_settings1 = RunSettings("echo", ["hello", "world"], run_command="srun")

    entity2 = Orchestrator("entity2")
    launch_settings2 = RunSettings("echo", ["spam", "eggs"], run_command="srun")

    pair1 = MPMDPair(entity1, launch_settings1)
    mpmd_pairs.append(pair1)
    mpmd_job = MPMDJob(mpmd_pairs)

    with pytest.raises(SSUnsupportedError) as ex:
        mpmd_job.add_mpmd_pair(entity2, launch_settings2)

        assert "MPMD pairs must all share the same entity type." in ex.value.args[0]


def test_create_mpmdjob_invalid_mpmdpairs():
    """Test that an error is raised when a pairs is added to an mpmd job that
    does not have the same launcher type"""

    mpmd_pairs = []
    entity1 = Model("entity1", None, None)
    launch_settings1 = RunSettings("echo", ["hello", "world"], run_command="srun")

    entity1 = Model("entity1", None, None)
    launch_settings2 = RunSettings("echo", ["spam", "eggs"], run_command="mpirun")

    pair1 = MPMDPair(entity1, launch_settings1)
    pair2 = MPMDPair(entity1, launch_settings2)

    mpmd_pairs.append(pair1)
    mpmd_pairs.append(pair2)

    with pytest.raises(SSUnsupportedError) as ex:
        MPMDJob(mpmd_pairs)
    assert "MPMD pairs must all share the same launcher." in ex.value.args[0]


def test_create_mpmdjob_valid_mpmdpairs():
    """Test that all pairs have the same entity type is enforced when creating an MPMDJob"""

    mpmd_pairs = []
    entity1 = Model("entity1", None, None)
    launch_settings1 = RunSettings("echo", ["hello", "world"], run_command="srun")
    entity1 = Model("entity1", None, None)
    launch_settings2 = RunSettings("echo", ["spam", "eggs"], run_command="srun")

    pair1 = MPMDPair(entity1, launch_settings1)
    pair2 = MPMDPair(entity1, launch_settings2)

    mpmd_pairs.append(pair1)
    mpmd_pairs.append(pair2)
    mpmd_job = MPMDJob(mpmd_pairs)

    assert isinstance(mpmd_job, MPMDJob)
