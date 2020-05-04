import zmq
import pickle
from subprocess import Popen, CalledProcessError, PIPE, run
from smartsim.launcher.shell import execute_async_cmd, execute_cmd
from smartsim.cmdSchema import RemoteRequest, RemoteResponse

from smartsim.utils import get_logger
logger = get_logger()

class CMDServer:

    def __init__(self, address, port):
        """Initialize a command server at a tcp address. The
           command server is used for executing commands on
           another management or login node that is connected
           to the same network

        :param address: IPv4 address of the node
        :type address: str
        :param port: port of the address
        :type port: int
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://" + address + ":" + str(port))
        self.running = False

    def serve(self):
        """Continually serve requests until a shutdown command is
           recieved.
        """
        self.running = True
        while self.running:
            request = self.socket.recv()
            remote_request = pickle.loads(request)
            logger.info("Got cmd: " + " ".join(remote_request.cmd))
            returncode, out, err = self.process_command(remote_request)
            response = RemoteResponse(returncode, out, err)
            rep = response.serialize()
            # send the response back to the compute node
            self.socket.send(rep)

    def process_command(self, remote_request):
        """Process a recieved command and direct to the function
           responsible for serving the request.

        :param remote_request: the RemoteRequest instance
        :type remote_request: RemoteRequest
        :return: returncode, output, error of command
        :rtype: tuple of (int, str, str)
        """
        cmd = remote_request.cmd
        if isinstance(cmd, list):
            cmd = remote_request.cmd[0]
        if cmd == "shutdown":
            return self.shutdown()
        elif cmd == "ping":
            return self.pong()
        else:
            return self.run_command(remote_request)

    def run_command(self, request):
        """Run an asynchronous or synchronous command using the
           shell library.

        :param request: the RemoteRequest instance
        :type request: RemoteRequest
        :return: returncode, output, error of the command
        :rtype: tuple of (int, str, str)
        """
        if request.is_async:
            return execute_async_cmd(request.cmd,
                                     request.cwd,
                                     remote=False,
                                     verbose=False)
        else:
            return execute_cmd(request.cmd,
                               shell=request.shell,
                               cwd=request.cwd,
                               proc_input=request.input,
                               timeout=request.timeout,
                               env=request.env,
                               remote=False,
                               verbose=False)

    def shutdown(self):
        """Shutdown the command server

        :return: placeholders to ack that server has been shutdown
        :rtype: tuple of (int, str, str)
        """
        self.running = False
        self.socket.close()
        self.context.term()
        return 0, "OK", ""

    def pong(self):
        """Reply to ensure client that server is setup.

        :return: placeholders to ack that server is live
        :rtype: tuple of (int, str, str)
        """
        return 0, "OK", ""



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--address', type=str, default="127.0.0.1",
                        help='Address of the command server')
    parser.add_argument('--port', type=int, default=5555,
                        help='Port of the command server')
    args = parser.parse_args()

    cmd_center = CMDServer(args.address, args.port)
    cmd_center.serve()