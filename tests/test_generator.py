from smartsim.settings.launchSettings import LaunchSettings
from smartsim.entity._new_ensemble import Ensemble
from smartsim.entity.model import Application
from smartsim._core.control.manifest import Manifest
from smartsim._core.generation.generator import Generator
import os

launch_settings = LaunchSettings("slurm")

# TODO remove run_settings and exe requirements
application = Application("app", exe="python",run_settings="RunSettings")
def test_generate_experiment_directory(test_dir):
    manifest = Manifest()
    print(os.getenv("SMARTSIM_LOG_LEVEL"))
    
    