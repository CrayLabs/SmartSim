
import numpy as np
import matplotlib.pyplot as plt

from smartredis import Client
from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.settings import RunSettings

exp = Experiment("finite_volume_simulation", launcher="local")
db = Orchestrator(port=6780)

# simulation parameters and plot settings
fig = plt.figure(figsize=(12,6), dpi=80)
time_steps, seed = 3000, 42

# define how simulation should be executed
settings = exp.create_run_settings("python",
                                   exe_args=["fv_sim.py",
                                             f"--seed={seed}",
                                             f"--steps={time_steps}"])
model = exp.create_model("fv_simulation", settings)

# tell exp.generate to include this file in the created run directory
model.attach_generator_files(to_copy="fv_sim.py")

# generate directories for output, error and results
exp.generate(db, model, overwrite=True)

# start the database and connect client to get data
exp.start(db)
client = Client(address="127.0.0.1:6780", cluster=False)

# start simulation without blocking so data can be analyzed in real time
exp.start(model, block=False, summary=True)

# poll until data is available
client.poll_key("cylinder", 200, 100)
cylinder = client.get_tensor("cylinder").astype(bool)

for i in range(0, time_steps, 5): # plot every 5th timestep
    client.poll_key(f"data_{i}", 10, 1000)
    dataset = client.get_dataset(f"data_{i}")
    ux, uy = dataset.get_tensor("ux"), dataset.get_tensor("uy")

    plt.cla()
    ux[cylinder], uy[cylinder] = 0, 0
    vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (
        np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
    )
    vorticity[cylinder] = np.nan
    cmap = plt.cm.get_cmap("bwr").copy()
    cmap.set_bad(color='black')
    plt.imshow(vorticity, cmap=cmap)
    plt.clim(-.1, .1)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    plt.pause(0.001)

# Save figure
plt.savefig('latticeboltzmann.png', dpi=240)
plt.show()

exp.stop(db)
