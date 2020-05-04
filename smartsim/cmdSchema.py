
import pickle

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
        :param timeout: timeout for waiting for output, defaults to None
        :type timeout: int, optional
        """
        self.cmd = cmd_list
        self.cwd = cwd
        self.shell = shell
        self.input = proc_input
        self.is_async = is_async
        self.env = env
        self.timeout = timeout

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

