from ..config import CONFIG
from ..entity import EntityList, Model
from ..error import SmartSimError, SSConfigError
from ..launcher.util.shell import execute_cmd
from ..settings import RunSettings, SrunSettings, SbatchSettings, AprunSettings, QsubBatchSettings
from ..utils import get_logger
from ..utils.helpers import expand_exe_path, get_env

import os
import time
import uuid
import re
logger = get_logger(__name__)

def delete_elements(dictionary, key_list):
    """Delete elements from a dictionary.
    :param dictionary: the dictionary from which the elements must be deleted.
    :type dictionary: dict
    :param key_list: the list of keys to delete from the dictionary.
    :type key: any
    """
    for key in key_list:
        if key in dictionary:
            del dictionary[key]
    
class RayCluster(EntityList):
    """Entity used to run a Ray cluster on a given number of hosts. Ray is launched on each host,
    and the first host is used to launch the head node.
    
    :param name: The name of the entity.
    :type name: str
    :param path: path to output, error, and configuration files
    :type path: str
    :param ray_port: Port at which the head node will be running.
    :type ray_port: int
    :param ray_num_cpus: Number of CPUs used by Ray
    :type ray_num_cpus: int
    :param workers: Number of workers (the total size of the cluster is ``workers``+1).
    :type workers: int
    :param run_args: Arguments to pass to launcher to specify details such as partition, time, and so on.
    :type run_args: dict[str,str]
    :param launcher: Name of launcher to use for starting the cluster.
    :type launcher: str
    :param alloc: ID of allocation to run on, only used if launcher is Slurm and allocation is
                  obtained with ``ray.slurm.get_allocation``
    :type alloc: int
    :param batch: Whether cluster should be launched as batch file
    :type batch: bool
    :param time: The walltime the cluster will be running for
    :type time: str
    """

    def __init__(self,
                 name,
                 path=".",
                 ray_port=6780,
                 ray_num_cpus=12,
                 workers=0,
                 run_args={},
                 launcher='local',
                 batch=False,
                 time="01:00:00",
                 alloc=None):
        self._workers = workers
        self._ray_port = ray_port
        self._ray_password = uuid.uuid4()
        self._launcher = launcher.lower()
        self._run_args = run_args
        self._ray_num_cpus = ray_num_cpus
        self.head_model = None
        self.worker_model = None
        self.alloc = None
        self.batch_settings = None
        self._alloc = alloc
        self._batch = batch
        self._hosts = None
        self._time = time
        super().__init__(name=name, path=path)
    
    @property
    def batch(self):
        return self._batch   

    def _update_worker_model(self):
        self.worker_model.update_run_settings()

    def _initialize_entities(self):
        self.head_model = RayHead(name="head", params=None,
                                  path=os.path.join(self.path, 'head'),
                                  ray_password = self._ray_password,
                                  ray_port=self._ray_port,
                                  launcher=self._launcher,
                                  run_args=self._run_args,
                                  ray_num_cpus=self._ray_num_cpus,
                                  alloc=self._alloc,
                                  batch=self.batch,
                                  time=self._time)

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
                                          batch=self.batch,
                                          time=self._time)
            self.entities.append(self.worker_model)


    def _get_ray_head_node_address(self):
        """Get the ray head node host address from the log file produced
        by the head process.

        :return: address of the head host
        :rtype: str
        """
        # We can rely on the file name, because we set it when we create
        # the head model
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
        
        
class RayHead(Model):
    """Ray head node model.
    
    :param name: The name of the entity.
    :type name: str
    :param params: model parameters for writing into configuration files.
    :type params: dict
    :param path: path to output, error, and configuration files
    :type path: str
    :param ray_password: Password used to connect to Redis by Ray
    :type ray_password: str
    :param ray_port: Port at which the head node will be running
    :type ray_port: int
    :param ray_num_cpus: Number of CPUs used by Ray
    :type ray_num_cpus: int
    :param run_args: Arguments to pass to launcher to specify details such as partition, time, and so on
    :type run_args: dict[str,str]
    :param launcher: Name of launcher to use for starting the cluster
    :type launcher: str
    :param alloc: ID of allocation to run on, only used if launcher is Slurm and allocation is
                  obtained with ``ray.slurm.get_allocation``
    :type alloc: int
    :param batch: Whether the head node should be launched through a batch file
    :type batch: bool
    :param time: The walltime the head node will be running for
    :type time: str
    """
    def __init__(self, 
                 name, 
                 params, 
                 path, 
                 ray_password, 
                 ray_port=6780, 
                 run_args={}, 
                 ray_num_cpus=12, 
                 launcher='local',
                 alloc=None,
                 batch=False,
                 time="01:00:00"):
        self._ray_port = ray_port
        self._run_args = run_args.copy()
        self._ray_password = ray_password
        self._ray_num_cpus = ray_num_cpus
        self._alloc = alloc
        self._launcher = launcher
        self.address = None
        self._batch = batch
        self._time = time
        
        self._build_run_settings()
        super().__init__(name, params, path, self.run_settings)
        
    @property
    def batch(self):
        return self._batch
    
    def _build_run_settings(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ray_args = [f"{dir_path}/raystarter.py"]
        ray_args += [f"--num-cpus={self._ray_num_cpus}"]
        ray_args += [f"--port={self._ray_port}"]
        ray_args += [f"--redis-password={self._ray_password}"]
        
        if self._launcher == 'slurm':
            self.run_settings = self._build_srun_settings(ray_args)
        elif self._launcher == 'pbs':
            self.run_settings = self._build_pbs_settings(ray_args)
        else:
            raise NotImplementedError("Only Slurm and PBS launchers are supported.")
            
        self.run_settings.set_walltime(self._time)
        self.run_settings.set_tasks(1)
        self.run_settings.set_tasks_per_node(1)
    
    def _build_pbs_settings(self, ray_args):
        if self.batch:
            self.batch_settings = QsubBatchSettings(nodes=1, ncpus=1, time=self._time)
        self._run_args["sync-output"] = None
        aprun_settings = AprunSettings("python", exe_args=" ".join(ray_args),
                                       run_args=self._run_args, expand_exe=True)

        return aprun_settings
    
    def _build_srun_settings(self, ray_args):
        batch_args = self._run_args
        batch_args["overcommit"] = None
        batch_args.update(self._run_args)
        delete_elements(batch_args, ["nodes", "ntasks-per-node"])
            
        if self.batch:
            self.batch_settings = SbatchSettings(
                nodes=1, time=self._time, batch_args=batch_args
            )
        else:
            delete_elements(batch_args, ["oversubscribe"])

        run_args = batch_args.copy()
        run_args["unbuffered"] = None
        srun_settings = SrunSettings("python", exe_args=" ".join(ray_args),
                                     run_args=run_args, expand_exe=False,
                                     alloc=self._alloc)
        return srun_settings
            

class RayWorker(Model):
    """Ray head node model.
    
    :param name: The name of the entity.
    :type name: str
    :param params: model parameters for writing into configuration files.
    :type params: dict
    :param path: path to output, error, and configuration files
    :type path: str
    :param ray_password: Password used to connect to Redis by Ray
    :type ray_password: str
    :param ray_port: Port at which the head node will be running
    :type ray_port: int
    :param ray_num_cpus: Number of CPUs used by Ray
    :type ray_num_cpus: int
    :param run_args: Arguments to pass to launcher to specify details such as partition, time, and so on
    :type run_args: dict[str,str]
    :param head_model: This cluster's head model's entity
    :type head_model: RayHead
    :param launcher: Name of launcher to use for starting the cluster
    :type launcher: str
    :param alloc: ID of allocation to run on, only used if launcher is Slurm and allocation is
                  obtained with ``ray.slurm.get_allocation``
    :type alloc: int
    :param batch: Whether the head node should be launched through a batch file
    :type batch: bool
    :param time: The walltime the head node will be running for
    :type time: str
    """
    def __init__(self, 
                 name, 
                 params, 
                 path, 
                 workers, 
                 ray_password, 
                 ray_port, 
                 ray_num_cpus, 
                 run_args,
                 head_model, 
                 launcher, 
                 alloc=None, 
                 batch=False, 
                 time="01:00:00"):
        self._run_args = run_args.copy()
        self._ray_port = ray_port
        self._ray_password = ray_password
        self._ray_num_cpus = ray_num_cpus
        self._launcher = launcher
        self._workers = workers
        self.head_model = head_model
        self._alloc = alloc
        self._batch = batch
        self._time = time
        
        self._build_run_settings()
        super().__init__(name, params, path, self.run_settings)
    
    @property
    def batch(self):
        return self._batch
    
    def update_run_settings(self):
        self.run_settings.add_exe_args([f"--address={self.head_model.address}:{self._ray_port}"])

    def _build_run_settings(self):
        ray_args = ["start"]
        ray_args += [f"--redis-password={self._ray_password}"]
        ray_args += [f"--num-cpus={self._ray_num_cpus}"]
        ray_args += ["--block"]
        if self._launcher == 'slurm':
            self.run_settings = self._build_srun_settings(ray_args)
        elif self._launcher == 'pbs':
            self.run_settings = self._build_pbs_settings(ray_args)
        else:
            raise NotImplementedError("Only Slurm and PBS launchers are supported.")
        
        if self._time:
            self.run_settings.set_walltime(self._time)
        self.run_settings.set_tasks(self._workers)
        self.run_settings.set_tasks_per_node(1)

    def _build_srun_settings(self, ray_args):
        batch_args = self._run_args
        delete_elements(batch_args, ["nodes", "ntasks-per-node"])

        if self.batch:
            self.batch_settings = SbatchSettings(
                nodes=self._workers, time=self._time, batch_args=batch_args
            )
        run_args = batch_args.copy()
        run_args["unbuffered"] = None
        if not self.batch and self._alloc is None:
            run_args["overcommit"] = None
        srun_settings =  SrunSettings("ray", exe_args=" ".join(ray_args),
                                      run_args=run_args, expand_exe=False,
                                      alloc=self._alloc)
        return srun_settings

    def _build_pbs_settings(self, ray_args):
        batch_args = self._run_args
        if self.batch:
            self.batch_settings = QsubBatchSettings(nodes=self._workers, ncpus=1, time=self._time)
        self._run_args["sync-output"] = None
        aprun_settings = AprunSettings("ray", exe_args=" ".join(ray_args),
                                       run_args=self._run_args, expand_exe=False)
        
        return aprun_settings