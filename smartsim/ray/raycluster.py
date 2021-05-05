from ..config import CONFIG
from ..entity import EntityList, Model
from ..error import SmartSimError, SSConfigError
from ..launcher.util.shell import execute_cmd
from ..settings import RunSettings, SrunSettings, SbatchSettings
from ..utils import get_logger
from ..utils.helpers import expand_exe_path, get_env

import os
import zmq
import uuid
logger = get_logger(__name__)

class RayCluster(EntityList):
    """User facing object like the Orchestrator

    3 ways to do this
    ------------------

    1. leave it like this, and come up with strategies
    for dynamically creating the right run settings types
         - benefits: launcher agnostic
         - downsides: not always clear where or how to specify laucnher settings

    or

    2. we can subclass this class like we did the Orchestrator

    SlurmRayCluster(port, workers, batch, hosts, run_command, account, time, alloc, etc)
    CobaltRayCluster(port, workers, batch, hosts, run_command, account, time, queue, etc)
    PBSRayCluster(port, workers, batch, hosts, run_command, account, time, queue, etc)
     - benefits:  Very clear where or how to specify laucnher settings
     - downsides: not launcher agnostic

    3. Create a subclass of Model for each launcher and then
    come up with a general way to supplying launcher settings to
    the model types. this is proabbly the hardest.
     - benefits: launcher agnostic
     - downsides: not particular clear, harder to implement, requires intermediatary settings

    largely, we can base this off what we did with the Orchestrators

    Open Questions:
     - how do we test this?
     - do we include this as an optional package?
     - can we re-use our redis instllation?
     -
    """

    def __init__(self, name, path, ray_port=6780, workers=1, launcher='local', run_args={}, ray_num_cpus=12):
        self._workers = workers
        self._ray_port = ray_port
        self._password = uuid.uuid4()
        self._launcher = launcher
        self._run_args = run_args
        self._ray_num_cpus = ray_num_cpus
        super().__init__(name=name, path=path)

    def start_ray_job(self, job_script):
        # use cmdserver client to send job to daemon thread
        """Start a Ray Job

        :param request: RemoteRequest instance
        :type request: RemoteRequest
        :raises CommandServerError: if communication with the remote
                                     server fails
        :return: returncode, out, err of the command
        :rtype: tuple of (int, str, str)
        """
        address = self._get_cmd_server_address() # should wait until its spun up
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


    def _initialize_entities(self):
        # We have one entity for the server processes (server+ray head+ray processes)
        # and one entity for the worker nodes
        self._build_run_settings()
        self.head_node_model = RayHead(name="head_node", params=None,
                                       path=os.path.join(self.path, 'head'),
                                       run_settings=self._head_run_settings)
        
        self.entities.append(self.head_node_model)
        
        self.worker_node_model = RayWorker(name="head_node", params=None,
                                           path=os.path.join(self.path, 'head'),
                                           run_settings=self._worker_run_settings,
                                           head_node=self.head_node_model)

    def _build_run_settings(self):
        # called in _initialize_entities to create run settings
        # for objects. if choosing method 2, this will be written
        # in each subclass
        if self._launcher == 'slurm':
            self._head_run_args = {"nodes": 1,
                                   "ntasks-per-node": 1, # Ray will take care of resources.
                                   "ntasks": 1,
                                   "cpus-per-task": self._ray_num_cpus,
                                   "oversubscribe": None,
                                   "overcommit": None,
                                   "time": "12:00:00",
                                   "unbuffered": None}
            dir_path = os.path.dirname(os.path.realpath(__file__))
            head_args = [os.path.join(dir_path, "rayserverstarter.py")]
            head_args += [f"--ray-num-cpus={self._ray_num_cpus}"]
            head_args += [f"--ray-port={self._ray_port}"]
            head_args += [f"--ray-password={self._password}"]
            head_args += [f"--zmq-port={6788}"]
            logger.info(" ".join(head_server_args))
            self._head_run_settings = SrunSettings("python", exe_args=" ".join(head_args),
                                                   run_args=self._head_run_args, expand_exe=False)
            
            self._worker_run_args = self._head_run_args.copy()
            self._worker_run_args["nodes"] = workers-1
            self._head_run_settings = SrunSettings("ray", exe_args=" ".join(head_args),
                                                   run_args=self._head_run_args, expand_exe=False)
            
            self.batch_settings = SbatchSettings(nodes=self._workers, time=self._run_args["time"])
        else:
            raise NotImplementedError
        

    def _get_ray_head_node_address(self):
        """Get the ray head node address.. this may be tricky

        We used env vars for the command server but thats probably
        not going to work here.

        could get it from teh launcher.... or it could be added
        after launch... not really sure what to do here.

        :raises CommandServerError: if command server has not been setup
        :return: address of the command server
        :rtype: str
        """
        head_log = os.path.join(self.head_node_model.path, "head_node.out")
        max_attempts = 60
        attempts = 0
        while not os.path.isfile(head_log):
            time.sleep(1)
            attempts += 1
            if attempts==max_attempts:
                raise RuntimeError("Could not find Ray cluster head address.")
                
        attempts = 0
        head_ip = None
        while head_ip is None:
            time.sleep(1)
            with open(head_log) as fp:
                line = fp.readline()
                while line:
                    plain_line = re.sub('\033\\[([0-9]+)(;[0-9]+)*m', '', line) 
                    if "Local node IP:" in plain_line:
                        matches=re.search(r'(?<=Local node IP: ).*', plain_line)
                        head_ip = matches.group()
                        print(f"Ray cluster's head is running at {head_ip}")
                    line = fp.readline()       
            attempts += 1
            if attempts==max_attempts:
                raise RuntimeError("Could not find Ray cluster head address.")
                
        self.head_node_model.address = head_ip

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
        pass


class RayHead(Model):
    """Technically using facing but users will never see it similar to DBNode"""
    def __init__(self, name, params, path, run_settings):
        super().__init__(name, params, path, run_settings)
        self.address = None


class RayWorker(Model):
    """Technically using facing but users will never see it similar to DBNode"""
    def __init__(self, name, params, path, run_settings, head_node):
        super().__init__(name, params, path, run_settings)
        self.head_node = head_node
