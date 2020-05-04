import os
import zmq
import pickle
from .cmdSchema import RemoteRequest
from .error import RemoteLauncherError

from smartsim.utils import get_logger
logger = get_logger()

class CmdClient:

    def __init__(self, timeout=50):
        """initalize a cmd client to send requests to the command server

        :param timeout: timeout for requests, defaults to 50
        :type timeout: int, optional
        """
        self.timeout = timeout * 1000

    def _get_remote_address(self):
        """Obtain the remote address set as an environment variable

        :raises RemoteLauncherError: if remote launcher has not been setup
        :return: address of the remote launcher
        :rtype: str
        """
        if not "SMARTSIM_REMOTE" in os.environ:
            raise RemoteLauncherError(
                "Command server has not been setup")
        return os.environ["SMARTSIM_REMOTE"]

    def create_remote_request(self, cmd_list, **kwargs):
        """Create a RemoteRequest object to send. Optional arguments
           can be specified as kwargs.

        :param cmd_list: the list of commands to execute
        :type cmd_list: list of str
        :return: RemoteRequest instance
        :rtype: RemoteRequest
        """
        request = RemoteRequest(cmd_list, **kwargs)
        return request

    def execute_remote_request(self, request):
        """Execute a remote request object on the remote command server

        :param request: RemoteRequest instance
        :type request: RemoteRequest
        :raises RemoteLauncherError: if communication with the remote
                                     server fails
        :return: returncode, out, err of the command
        :rtype: tuple of (int, str, str)
        """
        address = self._get_remote_address()
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        socket.setsockopt(zmq.LINGER, 0) # immediately fail if connection fails
        socket.connect(address)
        try:
            socket.send(request.serialize())

            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            if poller.poll(self.timeout):
                response = socket.recv()
                response = pickle.loads(response)
                return response.returncode, response.output, response.error
            else:
                raise RemoteLauncherError(
                    f"Communication failed with command server at {address}")
        except zmq.error.Again:
            raise RemoteLauncherError(
                    f"Communication failed with command server at {address}")
        finally:
            socket.close()
            context.term()
