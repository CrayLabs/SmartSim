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

import logging
import time
import typing as t
from itertools import product

import redis
from redis.cluster import ClusterNode, RedisCluster
from redis.exceptions import ClusterDownError, RedisClusterException
from smartredis import Client
from smartredis.error import RedisReplyError

from ...entity import FSModel, FSScript
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
    :param ports: List of ports for each hostname
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
    cmd += ["--cluster-replicas", "0", "--cluster-yes"]
    returncode, out, err = execute_cmd(cmd, proc_input="yes", shell=False)

    if returncode != 0:
        logger.error(out)
        logger.error(err)
        raise SSInternalError("Feature store '--cluster create' command failed")
    logger.debug(out)


def check_cluster_status(
    hosts: t.List[str], ports: t.List[int], trials: int = 10
) -> None:  # cov-wlm
    """Check that a Redis/KeyDB cluster is up and running

    :param hosts: List of hostnames to connect to
    :param ports: List of ports for each hostname
    :param trials: number of attempts to verify cluster status

    :raises SmartSimError: If cluster status cannot be verified
    """
    cluster_nodes = [
        ClusterNode(get_ip_from_host(host), port)
        for host, port in product(hosts, ports)
    ]

    if not cluster_nodes:
        raise SSInternalError(
            "No cluster nodes have been set for feature store status check."
        )

    logger.debug("Beginning feature store cluster status check...")
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


def fs_is_active(hosts: t.List[str], ports: t.List[int], num_shards: int) -> bool:
    """Check if a FS is running

    if the FS is clustered, check cluster status, otherwise
    just ping FS.

    :param hosts: list of hosts
    :param ports: list of ports
    :param num_shards: Number of FS shards
    :return: Whether FS is running
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


def set_ml_model(fs_model: FSModel, client: Client) -> None:
    logger.debug(f"Adding FSModel named {fs_model.name}")

    for device in fs_model.devices:
        try:
            if fs_model.is_file:
                client.set_model_from_file(
                    name=fs_model.name,
                    model_file=str(fs_model.file),
                    backend=fs_model.backend,
                    device=device,
                    batch_size=fs_model.batch_size,
                    min_batch_size=fs_model.min_batch_size,
                    min_batch_timeout=fs_model.min_batch_timeout,
                    tag=fs_model.tag,
                    inputs=fs_model.inputs,
                    outputs=fs_model.outputs,
                )
            else:
                if fs_model.model is None:
                    raise ValueError(f"No model attacted to {fs_model.name}")
                client.set_model(
                    name=fs_model.name,
                    model=fs_model.model,
                    backend=fs_model.backend,
                    device=device,
                    batch_size=fs_model.batch_size,
                    min_batch_size=fs_model.min_batch_size,
                    min_batch_timeout=fs_model.min_batch_timeout,
                    tag=fs_model.tag,
                    inputs=fs_model.inputs,
                    outputs=fs_model.outputs,
                )
        except RedisReplyError as error:  # pragma: no cover
            logger.error("Error while setting model on feature store.")
            raise error


def set_script(fs_script: FSScript, client: Client) -> None:
    logger.debug(f"Adding FSScript named {fs_script.name}")

    for device in fs_script.devices:
        try:
            if fs_script.is_file:
                client.set_script_from_file(
                    name=fs_script.name, file=str(fs_script.file), device=device
                )
            elif fs_script.script:
                if isinstance(fs_script.script, str):
                    client.set_script(
                        name=fs_script.name, script=fs_script.script, device=device
                    )
                elif callable(fs_script.script):
                    client.set_function(
                        name=fs_script.name, function=fs_script.script, device=device
                    )
            else:
                raise ValueError(f"No script or file attached to {fs_script.name}")
        except RedisReplyError as error:  # pragma: no cover
            logger.error("Error while setting model on feature store.")
            raise error


def shutdown_fs_node(host_ip: str, port: int) -> t.Tuple[int, str, str]:  # cov-wlm
    """Send shutdown signal to FS node.

    Should only be used in the case where cluster deallocation
    needs to occur manually. Usually, the SmartSim job manager
    will take care of this automatically.

    :param host_ip: IP of host to connect to
    :param ports: Port to which node is listening
    :return: returncode, output, and error of the process
    """
    redis_cli = CONFIG.database_cli
    cmd = [redis_cli, "-h", host_ip, "-p", str(port), "shutdown"]
    returncode, out, err = execute_cmd(cmd, proc_input="yes", shell=False, timeout=10)

    if returncode != 0:
        logger.error(out)
        err_msg = "Error while shutting down DB node. "
        err_msg += f"Return code: {returncode}, err: {err}"
        logger.error(err_msg)
    elif out:
        logger.debug(out)

    return returncode, out, err
