import os
import zmq
import pickle
from .cmdSchema import RemoteRequest
from ..error import CommandServerError

from smartsim.utils import get_logger
logger = get_logger()

class CmdClient:

    def __init__(self, timeout=50):
        """initalize a cmd client to send requests to the command server

        :param timeout: timeout for requests, defaults to 50
        :type timeout: int, optional
        """
        self.timeout = timeout * 1000

    def _get_cmd_server_address(self):
        """Obtain the command server address set as an environment variable

        :raises CommandServerError: if command server has not been setup
        :return: address of the command server
        :rtype: str
        """
        if not "SMARTSIM_REMOTE" in os.environ:
            raise CommandServerError(
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
        :raises CommandServerError: if communication with the remote
                                     server fails
        :return: returncode, out, err of the command
        :rtype: tuple of (int, str, str)
        """
        address = self._get_cmd_server_address()
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        socket.setsockopt(zmq.LINGER, 0) # immediately fail if connection fails
        socket.connect(address)
        try:
            socket.send(request.serialize())

            # set timeout
            timeout = self.timeout
            if request.timeout:
                timeout = request.timeout

            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)

            if poller.poll(timeout):
                response = socket.recv()
                response = pickle.loads(response)
                return response.returncode, response.output, response.error
            else:
                raise CommandServerError(
                    f"Communication failed with command server at {address}")
        except zmq.error.Again:
            raise CommandServerError(
                    f"Communication failed with command server at {address}")
        finally:
            socket.close()
            context.term()
