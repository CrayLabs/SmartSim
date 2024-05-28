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

from smartsim.database import FeatureStore
from smartsim.error import SmartSimError

try:
    from smartredis import Client

    config_setter = Client.config_set
except AttributeError:
    pytestmark = pytest.mark.skip(reason="SmartRedis version is < 0.3.1")

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


def test_config_methods(fsutils, prepare_fs, local_fs):
    """Test all configuration file edit methods on an active db"""
    fs = prepare_fs(local_fs).featurestore

    # test the happy path and ensure all configuration file edit methods
    # successfully execute when given correct key-value pairs
    configs = fsutils.get_fs_configs()
    for setting, value in configs.items():
        config_set_method = fsutils.get_config_edit_method(fs, setting)
        config_set_method(value)

    # ensure SmartSimError is raised when FeatureStore.set_fs_conf
    # is given invalid CONFIG key-value pairs
    ss_error_configs = fsutils.get_smartsim_error_fs_configs()
    for key, value_list in ss_error_configs.items():
        for value in value_list:
            with pytest.raises(SmartSimError):
                fs.set_fs_conf(key, value)

    # ensure TypeError is raised when FeatureStore.set_fs_conf
    # is given either a key or a value that is not a string
    type_error_configs = fsutils.get_type_error_fs_configs()
    for key, value_list in type_error_configs.items():
        for value in value_list:
            with pytest.raises(TypeError):
                fs.set_db_conf(key, value)


def test_config_methods_inactive(fsutils):
    """Ensure a SmartSimError is raised when trying to
    set configurations on an inactive feature store
    """
    fs = FeatureStore()
    configs = fsutils.get_fs_configs()
    for setting, value in configs.items():
        config_set_method = fsutils.get_config_edit_method(fs, setting)
        with pytest.raises(SmartSimError):
            config_set_method(value)
