import numpy as np
import matplotlib.pyplot as plt


def plot_lattice_vorticity(timestep, ux, uy, cylinder):
    
    fig = plt.figure(figsize=(12,6), dpi=80)

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
    print(f"Vorticity plot at timestep: {timestep}\n")