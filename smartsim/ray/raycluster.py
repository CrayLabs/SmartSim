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

class RayCluster(EntityList):
    """Entity used to run a Ray cluster on a given number of hosts. Ray is launched on each host,
    and the first host is used to launch the head node.
    
    :param name: The name of the entity.
    :type name: str
    :param path: Path where the entity is launched.
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
                  obtained with ``ray.slurm.get_allocation``.
    :type alloc: int
    :param batch: Whether cluster should be launched as batch file.
    :type batch: bool
    """

    def __init__(self, name, path, ray_port=6780, ray_num_cpus=12, workers=0, run_args={}, launcher='local', alloc=None, batch=False):
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
                                  batch=self.batch)

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
                                          batch=self.batch)
            self.entities.append(self.worker_model)


    def _get_ray_head_node_address(self):
        """Get the ray head node address.

        :return: address of the head host
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
        
        
class RayHead(Model):
    """Technically using facing but users will never see it similar to DBNode"""
    def __init__(self, name, params, path, ray_password, ray_port=6780, run_args={}, ray_num_cpus=12, launcher='local', alloc=None, batch=False):
        self._ray_port = ray_port
        self._run_args = run_args
        self._ray_password = ray_password
        self._ray_num_cpus = ray_num_cpus
        self._alloc = alloc
        self._launcher = launcher
        self.address = None
        self._batch = batch
        
        self._build_run_settings()
        super().__init__(name, params, path, self.run_settings)
        
    @property
    def batch(self):
        return self._batch
    
    def _build_run_settings(self):
#         ray_args = ["start"]
#         ray_args += ["--head"]
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ray_args = [f"{dir_path}/rayserverstarter.py"]
        ray_args += [f"--num-cpus={self._ray_num_cpus}"]
        ray_args += [f"--port={self._ray_port}"]
        ray_args += [f"--redis-password={self._ray_password}"]
#         ray_args += ["--block"]
        
        if self._launcher == 'slurm':
            self.run_settings = self._build_srun_settings(ray_args)
        elif self._launcher == 'pbs':
            self.run_settings = self._build_pbs_settings(ray_args)
        else:
            raise NotImplementedError("Only Slurm and PBS launchers are supported.")
    
    def _build_pbs_settings(self, ray_args):
        if self.batch:
            self.batch_settings = QsubBatchSettings(nodes=1, ncpus=1)
        return AprunSettings("python", exe_args=" ".join(ray_args),
                            run_args=self._run_args, expand_exe=True)
    
    def _build_srun_settings(self, ray_args):
            batch_args = {"nodes": 1,
                          "ntasks-per-node": 1, # Ray will take care of resources.
                          "ntasks": 1}
            # no alloc and no batch means that we are inside an allocation
            # we need to overcommit one node
            if self._alloc is None and not self.batch:
                batch_args["overcommit"] = None
            #TODO reject nodes, ntasks, ntasks-per-node, and so on.
            batch_args.update(self._run_args)
            if self.batch:
                self.batch_settings = SbatchSettings(
                    nodes=1, time=batch_args["time"], batch_args=batch_args
                )
            else:
                if "oversubscribe" in batch_args.keys():
                    del batch_args["oversubscribe"]
            
            if self._launcher == 'slurm':
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

    def _build_srun_settings(self, ray_args):
        batch_args = {"nodes": self._workers,
                      "ntasks-per-node": 1, # Ray will take care of resources.
                      "ntasks": self._workers,
                      "time": "12:00:00"}
        batch_args.update(self._run_args)

        if self.batch:
            self.batch_settings = SbatchSettings(
                nodes=self._workers, time=batch_args["time"], batch_args=batch_args
            )
        run_args = batch_args.copy()
        run_args["unbuffered"] = None
        if not self.batch and self._alloc is None:
            run_args["overcommit"] = None
        return SrunSettings("ray", exe_args=" ".join(ray_args),
                            run_args=run_args, expand_exe=False,
                            alloc=self._alloc)

    def _build_pbs_settings(self, ray_args):
        if self.batch:
            self.batch_settings = QsubBatchSettings(nodes=self._workers, ncpus=1)
        return AprunSettings("ray", exe_args=" ".join(ray_args),
                             run_args=self._run_args, expand_exe=False)