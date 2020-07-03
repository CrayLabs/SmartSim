import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from smartsim import Client

if __name__ == "__main__":

    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ranks", type=int, default=1)
    argparser.add_argument("--time", type=int, default=0)
    args = argparser.parse_args()

    n_ranks = args.ranks
    t_step = args.time

    client = Client(cluster=True)

    atom_id = []
    atom_type = []
    atom_x = []
    atom_y = []
    atom_z = []
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

    n_atoms = len(atom_id)
    for i in range(n_atoms):
        print(f"{atom_id[i]} {atom_type[i]} {atom_x[i]} {atom_y[i]} {atom_z[i]}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Atom position')
    ax.scatter(atom_x, atom_y, atom_z)
    plt.savefig('atom_position.pdf')
