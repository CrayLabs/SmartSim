import numpy as np
import matplotlib.pyplot as plt


def plot_lattice_vorticity(timestep, ux, uy, cylinder):
    fig = plt.figure(figsize=(12, 6), dpi=80)

    plt.cla()
    ux[cylinder], uy[cylinder] = 0, 0
    vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (
        np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
    )
    vorticity[cylinder] = np.nan
    cmap = plt.cm.get_cmap("bwr").copy()
    cmap.set_bad(color="black")
    plt.imshow(vorticity, cmap=cmap)
    plt.clim(-0.1, 0.1)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect("equal")
    ax.set_title(f"Vorticity plot at timestep {timestep}\n")
    plt.pause(0.001)


def plot_lattice_norm(timestep, u, cylinder):
    fig = plt.figure(figsize=(12, 6), dpi=80)

    plt.cla()

    u[cylinder] = np.nan
    cmap = plt.cm.get_cmap("jet").copy()
    cmap.set_bad(color="black")
    plt.contour(u, cmap=cmap)
    plt.clim(-0.1, 0.1)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect("equal")
    ax.set_title(f"Velocity magnitude at timestep {timestep}\n")
    plt.pause(0.001)


def plot_lattice_probes(timestep, probe_x, probe_y, probe_u):
    fig = plt.figure(figsize=(12, 6), dpi=80)

    plt.cla()
    cmap = plt.cm.get_cmap("binary").copy()
    cmap.set_bad(color="black")
    plt.quiver(
        probe_x,
        probe_y,
        probe_u[:, :, 0],
        probe_u[:, :, 1],
        np.linalg.norm(probe_u, axis=2),
        cmap=cmap,
        scale=7,
        pivot='mid',
        angles='uv',
        width=0.003
    )
    plt.clim(-0.1, 0.1)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect("equal")
    ax.set_xlim([0, 399])
    ax.set_ylim([0, 99])
    ax.set_title(f"Velocity field at timestep {timestep}\n")
    plt.pause(0.001)
