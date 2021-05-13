import zmq
import pickle
import shlex
import os
from smartsim.launcher.util.shell import execute_async_cmd, execute_cmd

from smartsim.utils import get_logger
logger = get_logger()

class RayServer:

    def __init__(self, address, zmq_port, ray_port, ray_password, ray_num_cpus):
        """Initialize a command server at a tcp address. The
           command server is used for executing commands on
           another management or login node that is connected
           to the same network

        :param address: IPv4 address of the node
        :type address: str
        :param port: port of the address
        :type port: int
        """
        self.address = address
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://" + address + ":" + str(zmq_port))
        self.running = False
        self.ray_port = ray_port
        self.ray_password = ray_password
        self.ray_num_cpus = ray_num_cpus

    def start(self):
        """Continually serve requests until a shutdown command is
           received.
        """
        head_code, head_out, head_err = self.start_ray_head_node()
        print(head_out)
        print(head_err)
        self.running = True
        logger.info(
            "Ray Server started. Ready to serve incoming requests...")
        while self.running:
            try:
                request = self.socket.recv()
                remote_request = pickle.loads(request)
                if not remote_request.is_async:
                    returncode, out, err = self.process_command(remote_request)
                else:
                    self.process_command(remote_request)
                    print("Command processed")
                    returncode = 0
                    out = "Process launched on head node"
                    err = ""
                    
                response = RemoteResponse(returncode, out, err)
                rep = response.serialize()
                # send the response back to the compute node
                self.socket.send(rep)
            except KeyboardInterrupt:
                self.running = False

        # close the socket and terminate the context if
        # we are no longer running
        logger.info("Shutting down Command Server...")
        self.socket.close()
        self.context.term()

    def start_ray_head_node(self):
        # assuming this is running locally on the remote node
        cmd = ["ray", "start", 
               "--head",
               f"--port={self.ray_port}",
               f"--node-ip-address={self.address}",
               f"--redis-password={self.ray_password}",
               f"--num-cpus={self.ray_num_cpus}"]

        return execute_cmd(cmd, shell=False)

    def stop_ray_head_node(self):
        # assuming this is running locally on the remote node
        cmd = ["ray", "stop"]
        return execute_cmd(cmd, shell=False)

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
            return self.run_ray_job(remote_request)

    def run_ray_job(self, request):
        """Run a ray job

        :param request: the RemoteRequest instance
        :type request: RemoteRequest
        :return: returncode, output, error of the command
        :rtype: tuple of (int, str, str)

        essentially this will start the ray job and
        then send back the return code/error/output

        we may just want to write error and output to
        file.
        """
        logger.info("CMD: " + request.cmd)

        return execute_async_cmd(shlex.split(request.cmd),
                        cwd=request.cwd)
        
#         return execute_cmd(shlex.split(request.cmd),
#                             shell=False,
#                             cwd=request.cwd,
#                             proc_input=request.input,
#                             timeout=request.timeout,
#                             env=request.env,
#                             is_async=True)

    def shutdown(self):
        """Shutdown the ray Server.

        :return: placeholders to ack that server has been shutdown
        :rtype: tuple of (int, str, str)
        """
        logger.info(
                "Received shutdown command from SmartSim experiment")
        self.running = False
        return self.stop_ray_head_node()

    def pong(self):
        """Reply to ensure client that server is setup.

        :return: placeholders to ack that server is live
        :rtype: tuple of (int, str, str)
        """
        logger.info(
                "Received initialization comand from SmartSim experiment")
        return 0, "OK", ""


class RayServerError(Exception):
    pass
    
# PULLED DIRECTLY FROM COMMAND SERVER
# we can leave this as a general way of sending commands
# or specialize it to ray.. idk what is better.
class RemoteRequest:

    def __init__(self, cmd_list, cwd=None, shell=True, proc_input="",
                 is_async=False, env=None, timeout=None):
        """Initialize a command request to be sent over zmq to
           a CmdServer instance.

        :param cmd_list: list of commands to execute
        :type cmd_list: list of str
        :param cwd: current working directory, defaults to None
        :type cwd: str, optional
        :param shell: run in system shell, defaults to True
        :type shell: bool, optional
        :param proc_input: input to the process, defaults to ""
        :type proc_input: str, optional
        :param is_async: run asychronously and don't capture output, defaults to False
        :type is_async: bool, optional
        :param env: environment to run request with,
                    defaults to None (current environment)
        :param timeout: timeout for waiting for output in seconds, defaults to None
        :type timeout: int, optional
        """
        self.cmd = cmd_list
        self.cwd = cwd
        self.shell = shell
        self.input = proc_input
        self.is_async = is_async
        self.env = env
        if timeout:
            self.timeout = int(timeout) * 1000
        else:
            self.timeout = 10000

    def serialize(self):
        request = pickle.dumps(self)
        return request

class RemoteResponse:

    def __init__(self, returncode, output, error):
        """Create a response to a RemoteRequest to send back over
           zmq to a CmdClient with the returncode, output, and
           error of the process.

        :param returncode: returncode of the process
        :type returncode: int
        :param output: standard output of the process
        :type output: str
        :param error: standard error of the process
        :type error: str
        """
        self.output = output
        self.error = error
        self.returncode = returncode

    def serialize(self):
        response = pickle.dumps(self)
        return response
