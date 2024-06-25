# Installing SmartSim on MMMHub's Young

## One-time setup of the environment

```
module purge
module load python3/3.11.4 gcc-libs/9.2.0 cmake/3.27.3
export CC=$(which gcc) CXX=$(which g++)
export PATH=/home/mmm1399/local/smartsim-build/bin:$PATH
python3 -m venv ~/smartsim_venv/
echo "module load python/3.11.4 gcc-libs/9.2.0" >> ~/smartsim_venv/bin/activate
source ~/smartsim_venv/bin/activate
```

As of 26 June 2024, support for SGE is available via the ``develop``
branch of SmartSim on Github.com. Until the next release, Young users
should install SmartSim via the following instructions:

```
pip install "smartsim[ml] @ git+https://github.com/CrayLabs/SmartSim.git"
smart build --torch_dir=/home/mmm1399/local/libtorch/share/cmake/Torch
```

or alternatively, if users want to build and install the examples shown
during the workshop, they should do the following instead:

```
git clone https://github.com/CrayLabs/SmartSim.git --branch mmmhub-young ~/Scratch/smartsim
cd ~/Scratch/smartsim
pip install .[ml]
smart build --torch_dir=/home/mmm1399/local/libtorch/share/cmake/Torch
```

## Building the SmartRedis library for C++ and Fortran applications

```
git clone https://github.com/CrayLabs/SmartRedis.git ~/smartredis
module purge
module load gcc-libs/9.2.0 cmake/3.27.3
export CC=$(which gcc) CXX=$(which g++)
cd ~/smartredis
make lib-with-fortran
```

## Building the Workshop Examples

```
module purge
module load python3/3.11.4 gcc-libs/9.2.0 cmake/3.27.3
export CC=$(which gcc) CXX=$(which g++)
cd ~/Scratch/smartsim/mmmhub-workshop-examples/proglets
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../
make install -j
```

## Environment for running a SmartSim experiment

```
module purge
module load python3/3.11.4 gcc-libs/9.2.0
export CC=$(which gcc) CXX=$(which g++)
source ~/smartsim_venv/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/smartredis/install/lib
```





