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

from smartsim.error import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")

try:
    from smartredis import Client

    config_setter = Client.config_set
except AttributeError:
    pytestmark = pytest.mark.skip(reason="SmartRedis version is < 0.3.1")


def test_config_methods_on_wlm_single(fsutils, prepare_db, single_db):
    """Test all configuration file edit methods on single node WLM db"""

    fs = prepare_fs(single_fs).featurestore
    # test the happy path and ensure all configuration file edit methods
    # successfully execute when given correct key-value pairs
    configs = fsutils.get_fs_configs()
    for setting, value in configs.items():
        logger.debug(f"Setting {setting}={value}")
        config_set_method = fsutils.get_config_edit_method(fs, setting)
        config_set_method(value)

    # ensure SmartSimError is raised when a clustered feature store's
    # FeatureStore.set_fs_conf is given invalid CONFIG key-value pairs
    ss_error_configs = fsutils.get_smartsim_error_fs_configs()
    for key, value_list in ss_error_configs.items():
        for value in value_list:
            with pytest.raises(SmartSimError):
                fs.set_fs_conf(key, value)

    # ensure TypeError is raised when a clustered feature store's
    # FeatureStore.set_fs_conf is given invalid CONFIG key-value pairs
    type_error_configs = fsutils.get_type_error_fs_configs()
    for key, value_list in type_error_configs.items():
        for value in value_list:
            with pytest.raises(TypeError):
                fs.set_fs_conf(key, value)


def test_config_methods_on_wlm_cluster(fsutils, prepare_fs, clustered_fs):
    """Test all configuration file edit methods on an active clustered db"""

    fs = prepare_fs(clustered_fs).featurestore
    # test the happy path and ensure all configuration file edit methods
    # successfully execute when given correct key-value pairs
    configs = fsutils.get_fs_configs()
    for setting, value in configs.items():
        logger.debug(f"Setting {setting}={value}")
        config_set_method = fsutils.get_config_edit_method(fs, setting)
        config_set_method(value)

    # ensure SmartSimError is raised when a clustered feature store's
    # FeatureStore.set_fs_conf is given invalid CONFIG key-value pairs
    ss_error_configs = fsutils.get_smartsim_error_fs_configs()
    for key, value_list in ss_error_configs.items():
        for value in value_list:
            with pytest.raises(SmartSimError):
                logger.debug(f"Setting {key}={value}")
                fs.set_fs_conf(key, value)

    # ensure TypeError is raised when a clustered feature store's
    # FeatureStore.set_fs_conf is given invalid CONFIG key-value pairs
    type_error_configs = fsutils.get_type_error_fs_configs()
    for key, value_list in type_error_configs.items():
        for value in value_list:
            with pytest.raises(TypeError):
                logger.debug(f"Setting {key}={value}")
                fs.set_fs_conf(key, value)
