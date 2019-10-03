
import subprocess
from launcher import SlurmLauncher

from os import getcwd, path

class DataBase:

    def __init__(self, conf="./smartsimdb.conf", path=None):
        self.launcher = SlurmLauncher.SlurmLauncher()
        self.conf = conf
        self.exe = "keydb-server"
        self.path = getcwd()
        self.pid = None
   
    def start_db(self, nodes=1, ppn=1, duration="1:00:00", port=6379):
        print("Starting Database")
        
        # run_command
        cmd = [" ".join((self.exe, self.conf, "--port", str(port)))]
        
        # database launch settings
        settings = dict()
        settings["output_file"] = path.join(self.path, "smartsimdb.out")
        settings["err_file"] = path.join(self.path, "smartsimdb.err")
        settings["nodes"] = nodes
        settings["ppn"] = ppn
        settings["duration"] = duration
        settings["cmd"] = cmd
        
        # make script and launch
        self.launcher.make_script(**settings, script_name="smartsimdb", clear_previous=True)
        self.pid = self.launcher.submit_and_forget(wd=self.path)
    
if __name__ == "__main__":
    db = DataBase()
    db.start_db()

