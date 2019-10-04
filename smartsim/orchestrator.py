
from smartsim import Controller
from smartsim import SmartSimModule, SmartSimNode
from .error import SSConfigError, SmartSimError

from os import path, getcwd, environ

class Orchestrator(SmartSimModule):
    """The Orchestrator is responsible for launching the DataBase and connecting
       the various user specified entities(SmartSimNodes, Models) to the correct
       endpoints
    """
       
    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)
        self.control = Controller(state)
        self.set_state("Orchestration")
    
        
    def orchestrate(self):
        """Launch the database and connect all nodes and models to their
           proper endpoints in the database
        """
        self.launch_db()
        #nodes = self.get_db_nodes() # not needed for now 
    
    def get_db_nodes(self):
        orc_job = self.control.get_job("orchestrator")
        orc_nodes = self.control.get_job_nodes(orc_job)
        return orc_nodes

    def launch_db(self, nodes=1, ppn=1, duration="1:00:00", port=6379):
        # TODO maybe change this to use kwargs
        conf_path = self._get_conf_path() + "smartsimdb.conf"
        exe_args = conf_path + " --port " + str(port)
        path = getcwd()

        # set run settings for the controller
        settings = {
            "executable": "keydb-server",
            "exe_args": exe_args,
            "launcher": "slurm",
            "run_command": "srun",
            "nodes": 1
        }
        
        self.state.create_target("Orchestrator")
        self.state.create_model("orchestrator", "Orchestrator", path=path)
        self.control.set_settings(new_settings=settings)
        self.control.start(target="Orchestrator")
        
    def _get_conf_path(self):
        try:
            path = environ["ORCCONFIG"]
            return path
        except KeyError:
            raise SSConfigError(self.get_state(), "SmartSim environment not set up!")
        
        

        
    
  
            
