# Smart-Sim Library

    A library of tools dedicated to accelerating the convergence of AI and numerical simulation models.
    
    
### Current Features

   - Rapid generation of model instances with custom configurations.
   - Model agnotic in regards to configuration file format.
   - multiple heirarchies of configuration allowing for extensive experimentation
   - Parallel model execution compatible with Slurm, PBS, Urika workload managers
   
### Setup

###### To Run Examples

   - Clone the git repository
   > git clone https://github.com/Spartee/Smart-Sim.git Smart-Sim
   - Set SmartSim env variables
   > cd Smart-Sim && source smartsim/scripts/setup_env.sh
   - Install Launcher
   > cd && git clone https://stash.us.cray.com/scm/ard/poseidon-launcher.git
   - Go in poseidon-launcher and setup
   > cd poseidon-launcher && source setenv.sh   # currently requires a patch as well
   - Go to clone of Smart-Sim and install dependencies
   > cd $SMARTSIMHOME/../ && pip install -r requirements.txt
   - Run the examples
   > cd smartsim && python3 run_ss.py
   

###### Using a new Model

### Simulation.toml

   
