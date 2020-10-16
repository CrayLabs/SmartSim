import os
import time
import pytest
from smartsim.utils import get_env, remove_env
from smartsim.remote import init_command_server, shutdown_command_server
from smartsim.remote.cmdServer import CMDServer
from smartsim.launcher.shell import execute_cmd
from smartsim.error import CommandServerError

from threading import Thread
# start the command server
def run_command_center():
    server = CMDServer("127.0.0.1", 5555)
    server.serve()

t = Thread(target=run_command_center, args=())
t.start()

def test_cmd_server_init():
    """test the setup of the command server. Will raise
       a CommandServerError if it fails to send a ping
       to a running CommandServer
    """
    init_command_server()
    assert("SMARTSIM_REMOTE" in os.environ)

def test_remote_cmd():
    """Test sending a remote command to the Command Server"""
    returncode, output, error = execute_cmd(["echo", "hello"])
    assert(output.strip() == "hello")
    assert(returncode == 0)

def test_cmd_server_shutdown():
    """Test shuting down the command server instance"""
    shutdown_command_server()
    assert("SMARTSIM_REMOTE" not in os.environ)

def test_shutdown_cmd_server_thread():
    """Join back the server thread"""
    t.join()

# ---- Error Handling -----------------------------

def test_bad_init():
    """Test when init is called but no cmd_server can be found"""
    with pytest.raises(CommandServerError):
        init_command_server()

def test_bad_shutdown():
    with pytest.raises(CommandServerError):
        shutdown_command_server()
