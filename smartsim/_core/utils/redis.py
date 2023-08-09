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

import logging
import redis
import time
import typing as t

from itertools import product
from redis.cluster import RedisCluster, ClusterNode
from redis.exceptions import ClusterDownError, RedisClusterException
from smartredis import Client
from smartredis.error import RedisReplyError

from ...entity import DBModel, DBScript
from ...error import SSInternalError
from ...log import get_logger
from ..config import CONFIG
from ..launcher.util.shell import execute_cmd
from .network import get_ip_from_host

logging.getLogger("rediscluster").setLevel(logging.WARNING)
logger = get_logger(__name__)


def create_cluster(hosts: t.List[str], ports: t.List[int]) -> None:  # cov-wlm
    """Connect launched cluster instances.

    Should only be used in the case where cluster initialization
    needs to occur manually which is not often.

    :param hosts: List of hostnames to connect to
    :type hosts: List[str]
    :param ports: List of ports for each hostname
    :type ports: List[int]
    :raises SmartSimError: if cluster creation fails
    """
    ip_list = []
    for host in hosts:
        ip_address = get_ip_from_host(host)
        for port in ports:
            address = ":".join((ip_address, str(port) + " "))
            ip_list.append(address)

    # call cluster command
    redis_cli = CONFIG.database_cli
    cmd = [redis_cli, "--cluster", "create"]
    cmd += ip_list
    cmd += ["--cluster-replicas", "0"]
    returncode, out, err = execute_cmd(cmd, proc_input="yes", shell=False)

    if returncode != 0:
        logger.error(out)
        logger.error(err)
        raise SSInternalError("Database '--cluster create' command failed")
    logger.debug(out)


def check_cluster_status(
    hosts: t.List[str], ports: t.List[int], trials: int = 10
) -> None:  # cov-wlm
    """Check that a Redis/KeyDB cluster is up and running

    :param hosts: List of hostnames to connect to
    :type hosts: List[str]
    :param ports: List of ports for each hostname
    :type ports: List[int]
    :param trials: number of attempts to verify cluster status
    :type trials: int, optional

    :raises SmartSimError: If cluster status cannot be verified
    """
    cluster_nodes = [
        ClusterNode(get_ip_from_host(host), port)
        for host, port in product(hosts, ports)
    ]

    if not cluster_nodes:
        raise SSInternalError(
            "No cluster nodes have been set for database status check."
        )

    logger.debug("Beginning database cluster status check...")
    while trials > 0:
        # wait for cluster to spin up
        time.sleep(5)
        try:
            redis_tester: "RedisCluster[t.Any]" = RedisCluster(
                startup_nodes=cluster_nodes
            )
            redis_tester.set("__test__", "__test__")
            redis_tester.delete("__test__")  # type: ignore
            logger.debug("Cluster status verified")
            return
        except (ClusterDownError, RedisClusterException, redis.RedisError):
            logger.debug("Cluster still spinning up...")
            trials -= 1
    if trials == 0:
        raise SSInternalError("Cluster setup could not be verified")


def db_is_active(hosts: t.List[str], ports: t.List[int], num_shards: int) -> bool:
    """Check if a DB is running

    if the DB is clustered, check cluster status, otherwise
    just ping DB.

    :param hosts: list of hosts
    :type hosts: list[str]
    :param ports: list of ports
    :type ports: list[int]
    :param num_shards: Number of DB shards
    :type num_shards: int
    :return: Whether DB is running
    :rtype: bool
    """
    # if single shard
    if num_shards < 2:
        host = hosts[0]
        port = ports[0]
        try:
            client = redis.Redis(host=host, port=port, db=0)
            if client.ping():
                return True
            return False
        except redis.RedisError:
            return False
    # if a cluster
    else:
        try:
            check_cluster_status(hosts, ports, trials=1)
            return True
        # we expect this to fail if the cluster is not active
        except SSInternalError:
            return False


def set_ml_model(db_model: DBModel, client: Client) -> None:
    logger.debug(f"Adding DBModel named {db_model.name}")

    for device in db_model.devices:
        try:
            if db_model.is_file:
                client.set_model_from_file(
                    name=db_model.name,
                    model_file=str(db_model.file),
                    backend=db_model.backend,
                    device=device,
                    batch_size=db_model.batch_size,
                    min_batch_size=db_model.min_batch_size,
                    tag=db_model.tag,
                    inputs=db_model.inputs,
                    outputs=db_model.outputs,
                )
            else:
                client.set_model(
                    name=db_model.name,
                    model=db_model.model,
                    backend=db_model.backend,
                    device=device,
                    batch_size=db_model.batch_size,
                    min_batch_size=db_model.min_batch_size,
                    tag=db_model.tag,
                    inputs=db_model.inputs,
                    outputs=db_model.outputs,
                )
        except RedisReplyError as error:  # pragma: no cover
            logger.error("Error while setting model on orchestrator.")
            raise error


def set_script(db_script: DBScript, client: Client) -> None:
    logger.debug(f"Adding DBScript named {db_script.name}")

    for device in db_script.devices:
        try:
            if db_script.is_file:
                client.set_script_from_file(
                    name=db_script.name, file=str(db_script.file), device=device
                )
            else:
                if isinstance(db_script.script, str):
                    client.set_script(
                        name=db_script.name, script=db_script.script, device=device
                    )
                else:
                    client.set_function(
                        name=db_script.name, function=db_script.script, device=device
                    )

        except RedisReplyError as error:  # pragma: no cover
            logger.error("Error while setting model on orchestrator.")
            raise error
