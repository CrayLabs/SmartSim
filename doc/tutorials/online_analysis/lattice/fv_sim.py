import numpy as np
from smartredis import Client, Dataset

def finite_volume_simulation(steps=4000, x_res=400, y_res=100,
                             rho0=100, tau=0.6, seed=42):
    """ Finite Volume simulation """
    client = Client() # Addresses passed to job through SmartSim

    # Lattice speeds / weights
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
    cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
    weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1

    # Initial Conditions
    F = np.ones((y_res, x_res, NL))
    np.random.seed(seed)
    F += 0.01 * np.random.randn(y_res, x_res,NL)
    X, Y = np.meshgrid(range(x_res), range(y_res))
    F[:,:,3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X/x_res * 4))
    rho = np.sum(F,2)
    for i in idxs:
        F[:,:,i] *= rho0 / rho

    # save cylinder location to database
    cylinder = (X - x_res/4)**2 + (Y - y_res/2)**2 < (y_res/4)**2 # bool array
    client.put_tensor("cylinder", cylinder.astype(np.int8))

    for time_step in range(steps): # simulation loop
        if time_step % 100 == 0:
            print(f"Step {time_step}", flush=True)
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
            F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)

        bndryF = F[cylinder,:]
        bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]

        rho = np.sum(F, 2)
        ux  = np.sum(F * cxs, 2) / rho
        uy  = np.sum(F * cys, 2) / rho

        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )
        F += -(1.0/tau) * (F - Feq)
        F[cylinder,:] = bndryF

        # send every 5 time_step to reduce memory consumption
        if time_step % 5 == 0:
            dataset = create_dataset(time_step, ux, uy)
            client.put_dataset(dataset)

    # send last time step to see final result
    dataset = create_dataset(time_step, ux, uy)
    client.put_dataset(dataset)


def create_dataset(time_step, ux, uy):
    """Create SmartRedis Dataset containing multiple NumPy arrays
    to be stored at a single key within the database"""
    dataset = Dataset(f"data_{time_step}")
    dataset.add_tensor("ux", ux)
    dataset.add_tensor("uy", uy)
    return dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Finite Volume Simulation")
    parser.add_argument('--steps', type=int, default=4000,
                        help='Number of simulation time steps')
    parser.add_argument('--seed', type=int, default=44,
                        help='Random Seed')
    args = parser.parse_args()
    finite_volume_simulation(steps=args.steps, seed=args.seed)