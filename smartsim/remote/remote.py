
import os

from .cmdClient import CmdClient
from ..error import CommandServerError
from ..utils import get_logger, remove_env
logger = get_logger(__name__)


def init_command_server(addr="127.0.0.1", port=5555):
    """Initialize a command server so that SmartSim can be
        run on compute nodes where slurm and other workload
        manager commands are not available. Command servers
        must be located on the same system, and be reachable
        via tcp.

        Keep in mind that this does not actually start the
        CmdServer instance, that must be started manually.

    :param addr: address of the Command Server launched seperately,
                    defaults to "127.0.0.1"
    :type addr: str, optional
    :param port: port of the Command Server, defaults to 5555
    :type port: int, optional
    :raises CommandServerError: If setup with remote launcher
                                    fails
    """
    # set the remote address for the cmd_client
    address = "tcp://" + ":".join((addr, str(port)))
    os.environ["SMARTSIM_REMOTE"] = address
    try:
        client = CmdClient()
        request = client.create_remote_request(["ping"], timeout=5)
        _, out, _ = client.execute_remote_request(request)
        logger.info("Command Server setup for remote commands")
    except CommandServerError as e:
        logger.error("Could not initialize Command Server")
        raise e

def shutdown_command_server():
    """Shutdown a command server launched by the user by
       sending a shutdown request to the running CmdServer.

    :raises CommandServerError: If shutdown fails
    """
    try:
        # ensure remote has been set in environment
        address = os.environ["SMARTSIM_REMOTE"]

        client = CmdClient()
        request = client.create_remote_request(["shutdown"], timeout=5)
        _, out, _ = client.execute_remote_request(request)
        logger.info("Command Server shutdown successful")

        # remove the remote env so future calls do not use this
        # command server
        remove_env("SMARTSIM_REMOTE")

    except KeyError:
        remove_env("SMARTSIM_REMOTE")
        logger.error("Failed to shutdown Command Server")
        raise CommandServerError("Could not find Command Center to shutdown")

    except CommandServerError as e:
        remove_env("SMARTSIM_REMOTE")
        logger.error("Failed to shutdown Command Server")
        raise e

