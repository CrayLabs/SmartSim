# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
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

import redis
from rediscluster import RedisCluster
from rediscluster.exceptions import ClusterDownError, RedisClusterException

logging.getLogger("rediscluster").setLevel(logging.WARNING)

from ...error import SSInternalError
from ...log import get_logger
from ..config import CONFIG
from ..launcher.util.shell import execute_cmd

logger = get_logger(__name__)


def create_cluster(hosts, ports):  # cov-wlm
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
        ip = get_ip_from_host(host)
        for port in ports:
            address = ":".join((ip, str(port) + " "))
            ip_list.append(address)

    # call cluster command
    redis_cli = CONFIG.redis_cli
    cmd = [redis_cli, "--cluster", "create"]
    cmd += ip_list
    cmd += ["--cluster-replicas", "0"]
    returncode, out, err = execute_cmd(cmd, proc_input="yes", shell=False)

    if returncode != 0:
        logger.error(out)
        logger.error(err)
        raise SSInternalError("Database '--cluster create' command failed")
    logger.debug(out)


def check_cluster_status(hosts, ports, trials=10):  # cov-wlm
    """Check that a Redis/KeyDB cluster is up and running

    :param hosts: List of hostnames to connect to
    :type hosts: List[str]
    :param ports: List of ports for each hostname
    :type ports: List[int]
    :param trials: number of attempts to verify cluster status
    :type trials: int, optional

    :raises SmartSimError: If cluster status cannot be verified
    """
    host_list = []
    for host in hosts:
        for port in ports:
            host_dict = dict()
            host_dict["host"] = get_ip_from_host(host)
            host_dict["port"] = port
            host_list.append(host_dict)

    logger.debug("Beginning database cluster status check...")
    while trials > 0:
        # wait for cluster to spin up
        time.sleep(5)
        try:
            redis_tester = RedisCluster(startup_nodes=host_list)
            redis_tester.set("__test__", "__test__")
            redis_tester.delete("__test__")
            logger.debug("Cluster status verified")
            return
        except (ClusterDownError, RedisClusterException, redis.RedisError):
            logger.debug("Cluster still spinning up...")
            trials -= 1
    if trials == 0:
        raise SSInternalError("Cluster setup could not be verified")
