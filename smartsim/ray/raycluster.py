from ..config import CONFIG
from ..entity import EntityList, Model
from ..error import SmartSimError, SSConfigError
from ..launcher.util.shell import execute_cmd
from ..settings import RunSettings, SrunSettings, SbatchSettings
from ..utils import get_logger
from ..utils.helpers import expand_exe_path, get_env
from .rayserver import RemoteRequest, RemoteResponse, RayServerError

import os
import time
import uuid
import re
import zmq
import pickle
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

    def __init__(self, name, path, ray_port=6780, ray_num_cpus=12, workers=0, zmq_port=7599, run_args={}, launcher='local', alloc=None, batch=False):
        self._workers = workers
        self._ray_port = ray_port
        self._ray_password = uuid.uuid4()
        self._launcher = launcher
        self._run_args = run_args
        self._ray_num_cpus = ray_num_cpus
        self._zmq_port = zmq_port
        self.head_model = None
        self.worker_model = None
        self.alloc = None
        self.batch_settings = None
        self.timeout = 100000
        self._alloc = alloc
        self._batch = batch
        super().__init__(name=name, path=path)

    def batch(self):
        return self._batch
        
    def start_ray_job(self, job_script, script_args=None):
        # use cmdserver client to send job to daemon thread
        """Start a Ray Job

        :param job_script: path to script to submit to Ray cluster. It can
                           be a Python or a Yaml file.
        :type job_script: str
        :param script_args: string with arguments to pass to ``job_script`` (used only
                            if the script is a Python script).
        :type script_args: str
        :raises CommandServerError: if communication with the remote
                                     server fails
        :return: returncode, out, err of the command
        :rtype: tuple of (int, str, str)
        """
        
        if job_script.endswith(".py"):
            cmd_list = ["python", job_script]
            if script_args:
                cmd_list += [script_args]
            cmd_list += [f"--redis-password={self._ray_password}"]
            cmd_list += [f"--ray-address={self.head_model.address}:{self._ray_port}"]
            request = self._create_remote_request(" ".join(cmd_list))
        
        self.execute_remote_request(request)        

    def _update_worker_model(self):
        self.worker_model.update_run_settings()

    def _initialize_entities(self):
        self.head_model = RayHead(name="head", params=None,
                                  path=os.path.join(self.path, 'head'),
                                  ray_password = self._ray_password,
                                  ray_port=self._ray_port,
                                  launcher=self._launcher,
                                  zmq_port=self._zmq_port,
                                  run_args=self._run_args,
                                  ray_num_cpus=self._ray_num_cpus,
                                  alloc=self._alloc,
                                  batch=self.batch())

        self.entities.append(self.head_model)

        if self._workers > 0:
            self.worker_model = RayWorker(name="workers", params=None,
                                          path=os.path.join(self.path, 'workers'),
                                          run_args=self._run_args,
                                          workers=self._workers,
                                          ray_port=self._ray_port,
                                          ray_password=self._ray_password,
                                          ray_num_cpus=self._ray_num_cpus,
                                          head_model=self.head_model,
                                          launcher=self._launcher,
                                          alloc=self._alloc,
                                          batch=self.batch())
            self.entities.append(self.worker_model)


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
        head_log = os.path.join(self.head_model.path, "head.out")
        
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
                    line = fp.readline()       
            attempts += 1
            if attempts==max_attempts:
                raise RuntimeError("Could not find Ray cluster head address.")
                
        self.head_model.address = head_ip
        
    def _get_cmd_server_address(self):
        return "tcp://" + self.head_model.address + ":" + str(self._zmq_port)

    def _create_remote_request(self, cmd_list, **kwargs):
        """Create a RemoteRequest object to send. Optional arguments
           can be specified as kwargs.

        :param cmd_list: the list of commands to execute
        :type cmd_list: list of str
        :return: RemoteRequest instance
        :rtype: RemoteRequest
        """
        request = RemoteRequest(cmd_list, timeout=100000, is_async=True, **kwargs)
        return request

    def execute_remote_request(self, request):
        address = self._get_cmd_server_address() # should wait until its spun up
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.SNDTIMEO, 10000)
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
                raise RayServerError(
                    f"Communication failed with command server at {address}")
        except zmq.error.Again:
            raise RayServerError(
                    f"Communication failed with command server at {address}")
        finally:
            socket.close()
            context.term()


class RayHead(Model):
    """Technically using facing but users will never see it similar to DBNode"""
    def __init__(self, name, params, path, ray_password, ray_port=6780, zmq_port=7599, run_args={}, ray_num_cpus=12, launcher='local', alloc=None, batch=False):
        self._ray_port = ray_port
        self._zmq_port = zmq_port
        self._run_args = run_args
        self._ray_password = ray_password
        self._ray_num_cpus = ray_num_cpus
        self._alloc = alloc
        self._launcher = launcher
        self.address = None
        self._batch = batch
        
        self._build_run_settings()
        super().__init__(name, params, path, self.run_settings)
        
    def batch(self):
        return self._batch
    
    def _build_run_settings(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ray_args = [os.path.join(dir_path, "rayserverstarter.py")]
        ray_args += [f"--ray-num-cpus={self._ray_num_cpus}"]
        ray_args += [f"--ray-port={self._ray_port}"]
        ray_args += [f"--ray-password={self._ray_password}"]
        ray_args += [f"--zmq-port={self._zmq_port}"]
        
        if self._launcher == 'slurm':
            self.run_settings = self._build_srun_settings(ray_args)
        else:
            raise NotImplementedError("Only Slurm launcher is supported.")
        
    def _build_srun_settings(self, ray_args):
    
            batch_args = {"nodes": 1,
                          "ntasks-per-node": 1, # Ray will take care of resources.
                          "ntasks": 1,
                          "cpus-per-task": self._ray_num_cpus,
                          "oversubscribe": None,
                          "overcommit": None,
                          "time": "12:00:00"}
            #have to include user-provided run args, but rejecting nodes, ntasks, ntasks-per-node, and so on.
            if self.batch():
                self.batch_settings = SbatchSettings(
                    nodes=1, time=batch_args["time"], batch_args=batch_args
                )
            run_args = batch_args.copy()
            run_args["unbuffered"] = None
            return SrunSettings("python", exe_args=" ".join(ray_args),
                                run_args=run_args, expand_exe=False,
                                alloc=self._alloc)

class RayWorker(Model):
    """Technically using facing but users will never see it similar to DBNode"""
    def __init__(self, name, params, path, run_args, workers, ray_port, ray_password, ray_num_cpus, head_model, launcher, alloc=None, batch=False):
        self._run_args = run_args
        self._ray_port = ray_port
        self._ray_password = ray_password
        self._ray_num_cpus = ray_num_cpus
        self._launcher = launcher
        self._workers = workers
        self.head_model = head_model
        self._alloc = alloc
        self._batch = batch
        
        self._build_run_settings()
        super().__init__(name, params, path, self.run_settings)
    
    def batch(self):
        return self._batch
    
    def update_run_settings(self):
        self.run_settings.add_exe_args([f"--address={self.head_model.address}:{self._ray_port}"])

    def _build_run_settings(self):
        ray_args = ["start"]
        ray_args += [f"--num-cpus={self._ray_num_cpus}"]
        ray_args += [f"--redis-password={self._ray_password}"]
        ray_args += ["--block"]
        if self._launcher == 'slurm':
            self.run_settings = self._build_srun_settings(ray_args)
            
        else:
            raise NotImplementedError("Only slurm launcher is supported.")

    def _build_srun_settings(self, ray_args):
        batch_args = {"nodes": self._workers,
                      "ntasks-per-node": 1, # Ray will take care of resources.
                      "ntasks": self._workers,
                      "cpus-per-task": self._ray_num_cpus,
                      "oversubscribe": None,
                      "overcommit": None,
                      "time": "12:00:00"}

        if self.batch():
            self.batch_settings = SbatchSettings(
                nodes=self._workers, time=batch_args["time"], batch_args=batch_args
            )
        run_args = batch_args.copy()
        run_args["unbuffered"] = None
        return SrunSettings("ray", exe_args=" ".join(ray_args),
                            run_args=run_args, expand_exe=False,
                            alloc=self._alloc)
