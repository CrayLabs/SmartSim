import zmq
import pickle
from subprocess import Popen, CalledProcessError, PIPE, run

from smartsim.utils import get_logger
logger = get_logger()

class CMDServer:

    def __init__(self, address, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://" + address + ":" + str(port))

    def serve(self):
        """Accept connections and listen on socket until shutdown command
           is recieved
        """
        shutdown = False
        while not shutdown:
            message = self.socket.recv()
            cmd, shell, cmd_input, cwd, is_asnyc = pickle.loads(message)
            logger.info("Got cmd: " + " ".join(cmd))

            if cmd[0] == "SHUTDOWN":
                shutdown = True
                self.socket.close()
                self.context.term()
                break

            if cmd[0] == "PING":
                rep = "OK"

            elif is_asnyc:
                rep = self.execute_asynch_command(cmd, cwd)

            elif cmd_input:
                returncode, out, err = self.execute_cmd_with_input(
                    cmd,
                    cmd_input,
                    shell=shell
                )
                rep = (returncode, out, err)

            else:
                returncode, out, err = self.execute_cmd(cmd, cwd=cwd, shell=shell)
                rep = (returncode, out, err)

            # send the response back to the compute node
            rep = pickle.dumps(rep)
            self.socket.send(rep)


    def execute_asynch_command(self, cmd, cwd):
        """Execute a command and do not wait for output"""
        try:
            popen_obj = Popen(cmd, cwd=cwd, shell=True)
        except OSError as err:
            logger.error(err)
            return -1
        return 1

    def execute_cmd(self, cmd_list, shell=False, cwd=None, verbose=False):
        """
            This is the function that runs a shell command and returns the output
            It will raise exceptions if the commands did not run successfully

            :param cmd_list: The command to be excuted
            :type cmd_list: List of str, optional str
            :param shell: The shell argument (which defaults to False)
                    specifies whether to use the shell as the program to execute.
                    If shell is True, it is recommended to pass args
                    as a string rather than as a sequence.
            :param cwd: The current working directory
            :type cwd: str
            :param verbose: Boolean for verbose output
            :type verbose: bool
            :returns: tuple of process returncode, output and error messages
        """
        # spawning the subprocess and connecting to its output
        proc = Popen(cmd_list, stdout=PIPE, stderr=PIPE, shell=shell, cwd=cwd)
        try:
            # waiting for the process to terminate and capture its output
            out, err = proc.communicate()
        except CalledProcessError as e:
            logger.error("Exception while attempting to start a shell process")
            err = f"Command: {proc.cmd} failed during remote execution"
            return proc.returncode, "", err

        # decoding the output and err and return as a string tuple
        return proc.returncode, out.decode('utf-8'), err.decode('utf-8')

    def execute_cmd_with_input(self, cmd_list, cmd_input, shell=True, encoding="utf-8"):
        """Run a command using the subprocess.run with input. Default to executing with
           the shell and utf-8 encoding.

           :param cmd_list: cmd to run
           :type cmd_list: list of str
           :param str cmd_input: input for the run command after cmd execution
           :param bool shell: toggle run in shell
           :param str encoding: encoding for the parsed process information
           :returns: tuple of process returncode, output and error messages
        """
        if isinstance(cmd_list, str):
            cmd_list = [cmd_list]

        proc = run(cmd_list,
                input=cmd_input,
                encoding=encoding,
                capture_output=True,
                shell=shell)
        return proc.returncode, proc.stdout, proc.stderr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--address', type=str, default="127.0.0.1",
                        help='Address of the command center')
    parser.add_argument('--port', type=int, default=5555,
                        help='Port of the command center')
    args = parser.parse_args()

    cmd_center = CMDServer(args.address, args.port)
    cmd_center.serve()