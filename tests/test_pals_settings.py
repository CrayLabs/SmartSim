import pytest

import sys

from smartsim.error import SSUnsupportedError
from smartsim.settings import PalsMpiexecSettings

default_exe = sys.executable
default_kwargs = {
    'fail_if_missing_exec':False
}

# Uncomment when 
#@pytest.mark.parametrize(
#    "function_name",[
#        'set_task_map',
#        'set_cpus_per_task',
#        'set_quiet_launch',
#        'set_walltime'
#    ]
#)
#def test_unsupported_methods(function_name):
#    settings = PalsMpiexecSettings(default_exe, **default_kwargs)
#    func = getattr(settings, function_name)
#    with pytest.raises(SSUnsupportedError):
#        func(None)

def test_cpu_binding_type():
    settings = PalsMpiexecSettings(default_exe, **default_kwargs)
    settings.set_cpu_binding_type('numa')
    assert settings.format_run_args() == ['--cpu-bind', 'numa']

def test_tasks_per_node():
    settings = PalsMpiexecSettings(default_exe, **default_kwargs)
    settings.set_tasks_per_node(48)
    assert settings.format_run_args() == ['--ppn', '48']

def test_broadcast():
    settings = PalsMpiexecSettings(default_exe, **default_kwargs)
    settings.set_broadcast()
    assert settings.format_run_args() == ['--transfer']

def test_format_env_vars():
    example_env_vars = {
        'FOO_VERSION':'3.14',
        'PATH':None,
        'LD_LIBRARY_PATH':None
    }
    settings = PalsMpiexecSettings(
        default_exe,
        **default_kwargs,
        env_vars=example_env_vars
    )
    formatted = ' '.join(settings.format_env_vars())
    expected = '--env FOO_VERSION=3.14 --envlist PATH,LD_LIBRARY_PATH'
    assert formatted == expected