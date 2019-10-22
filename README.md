# Smart-Sim Library

    A library of tools dedicated to accelerating the convergence of AI and numerical simulation models.
    
    
## Current Features

   - Rapid generation of model instances with custom configurations.
   - Model agnostic in regards to configuration file format.
   - Multiple hierarchies of configuration allowing for extensive experimentation
   - Parallel model execution compatible with Slurm workload manager
   - Online training and Inference
   - Model Parameter Optimization(MPO)
   
## Setup

   - Clone the git repository
      > git clone https://github.com/Spartee/Smart-Sim.git Smart-Sim
   - Set SmartSim env variables and add to python path
      > cd Smart-Sim && source setup_env.sh
   - Install Dependencies
      > pip install -r requirements.txt
   - Install Launcher
      > cd && git clone https://stash.us.cray.com/scm/ard/poseidon-launcher.git
   - Go in poseidon-launcher and setup
      > cd poseidon-launcher && source setenv.sh   # currently requires a patch as well
   - Install KeyDB
      > git clone https://github.com/JohnSully/KeyDB && cd KeyDB && make
      > cd src && export PATH=$PATH:$(pwd) # or add to .bashrc/.zshrc

## Run LAMMPS example

   - Go to the LAMMPS example folder
      > cd /examples/LAMMPS/
   - Run the LAMMPS example script
      > python run.py


   
