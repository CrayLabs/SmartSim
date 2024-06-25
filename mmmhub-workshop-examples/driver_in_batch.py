import argparse
import pathlib

from smartsim import Experiment

def main(args):
    # Define the top-level SmartSim object
    exp = Experiment("example_experiment", launcher="sge")

    # Create a SmartSim representative of the database
    db = exp.create_database(interface="ib0", db_nodes=1)

    # Define the settings to run a simulation with
    simulation_run_settings = exp.create_run_settings(
        exe=str(pathlib.Path.cwd()/"proglets"/"bin"/"smartredis_put_get_3D_cpp_parallel"),
        run_command="mpirun"
    )
    simulation_run_settings.set_nodes(1) # Use one node
    simulation_run_settings.set_tasks(4) # Use 4 MPI tasks

    # Create a SmartSim representative of a numerical model
    simulation = exp.create_model("example_simulation", simulation_run_settings)
    simulation.attach_generator_files(
        to_copy=["data/data_processing_script.txt"],
        to_symlink=["data/mnist_cnn.pt", "data/one.raw"]
    )

    # Create the run directories for the experiment
    exp.generate(db, simulation, overwrite=True)

    if not args.generate_only:
        # Start all the elements of the workflow and wait until the simulation finishes
        exp.start(db, simulation, block=True, summary=True)
        exp.stop(db)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample data from MFIX-exa simulation")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="If specified, preview the workflow without launching",
    )
    args = parser.parse_args()
    main(args)