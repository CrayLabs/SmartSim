import argparse
import pathlib

from smartsim import Experiment

# Define the top-level SmartSim object
exp = Experiment("hello_world", launcher="sge")

batch_settings = exp.create_batch_settings()
batch_settings.set_walltime("00:05:00")
batch_settings.set_account("Brunel_allocation")
batch_settings.set_project("Gold")
batch_settings.set_pe_type("mpi")
batch_settings.set_ncpus(1)

# Define the settings to run a perroquet with
perroquet_run_settings = exp.create_run_settings(
    exe="echo",
    exe_args=["Hello", "World!"],
    run_command="mpirun"
)
perroquet_run_settings.set_tasks(1)

# Create a SmartSim representative of a numerical model
perroquet = exp.create_model(
    "hello_world",
    perroquet_run_settings,
    batch_settings=batch_settings
)
exp.start(perroquet, block=True, summary=True)