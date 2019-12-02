# Smart-Sim Library

    A library of tools dedicated to accelerating the convergence of AI and numerical simulation models.

## Current Features

   - Online simulation interaction (Analysis, Visualization, Training, and Inference)
   - Model Parameter Optimization (MPO)
   - Compatiable with most simulation models
   - Rapid, programmatic generation and execution of model instances
   - User supplied model configuration generation strategies

## Setup

   - Clone the git repository
      > git clone https://github.com/Spartee/Smart-Sim.git Smart-Sim
   - Set SmartSim env variables and add to python path
      > cd Smart-Sim && source setup_env.sh
   - Install Dependencies
      > pip install -r requirements.txt
   - Install KeyDB
      > git clone https://github.com/JohnSully/KeyDB && cd KeyDB && make
      > cd src && export PATH=$PATH:$(pwd) # or add to .bashrc/.zshrc

## Run LAMMPS example

   - Go to the LAMMPS example folder
      > cd /examples/LAMMPS/
   - Run the LAMMPS example script
      > python run.py
