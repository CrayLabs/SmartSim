import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from smartsim import Client

if __name__ == "__main__":

    # The command line argument "ranks" is used to
    # know how many MPI ranks were used to run the
    # LAMMPS simulation because each MPI rank will send
    # a unique key to the database.  This command line
    # argument is provided programmatically as a
    # run setting in the SmartSim experiment script.
    # Similarly, the command line argument "time"
    # is used to set which time step data will be
    # pulled from the database.  This is also set
    # programmatically as a run setting in the SmartSim
    # experiment script
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ranks", type=int, default=1)
    argparser.add_argument("--time", type=int, default=0)
    args = argparser.parse_args()

    n_ranks = args.ranks
    t_step = args.time

    # Initialize the SmartSim client object and indicate
    # that a database cluster is being used with
    # cluster = True
    client = Client(cluster=True)

    # Create empty lists that we will fill with simulation data
    atom_id = []
    atom_type = []
    atom_x = []
    atom_y = []
    atom_z = []

    # We will loop over MPI ranks and fetch the data
    # associated with each MPI rank at a given time step.
    # Each variable is saved in a separate list.
    for i in range(n_ranks):
        key = f"atoms_rank_{i}_tstep_{t_step}_atom_id"
        print(f"loking for key {key}")
        atom_id.extend(client.get_array_nd_int64(key, wait=True))
        key = f"atoms_rank_{i}_tstep_{t_step}_atom_type"
        atom_type.extend(client.get_array_nd_int64(key, wait=True))
        key = f"atoms_rank_{i}_tstep_{t_step}_atom_x"
        atom_x.extend(client.get_array_nd_float64(key, wait=True))
        key = f"atoms_rank_{i}_tstep_{t_step}_atom_y"
        atom_y.extend(client.get_array_nd_float64(key, wait=True))
        key = f"atoms_rank_{i}_tstep_{t_step}_atom_z"
        atom_z.extend(client.get_array_nd_float64(key, wait=True))

    # We print the atom position data to check the accuracy of our results.
    # The printed data will be piped by SmartSim to an output file
    # in the experiment directory.
    n_atoms = len(atom_id)
    for i in range(n_atoms):
        print(f"{atom_id[i]} {atom_type[i]} {atom_x[i]} {atom_y[i]} {atom_z[i]}")

    # We plot the atom positions to check that the atom position distribution
    # is uniform, as expected.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Atom position')
    ax.scatter(atom_x, atom_y, atom_z)
    plt.savefig('atom_position.pdf')
