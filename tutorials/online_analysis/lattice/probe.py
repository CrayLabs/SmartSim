# import torch
# import torch.nn.functional as F
def multi_unsqueeze(tensor, axes: List[int]):
    for axis in axes:
        tensor = torch.unsqueeze(tensor, axis)

    return tensor

def probe_points(ux, uy, probe_x, probe_y, cylinder):
    ux[cylinder>0] = 0.0
    uy[cylinder>0] = 0.0
    ux = multi_unsqueeze(ux, [0, 0])
    uy = multi_unsqueeze(uy, [0, 0])
    # Explicit Euler scheme
    probe_xy = multi_unsqueeze(torch.stack((probe_x/200 - 1, probe_y/50 - 1), 2), [0])
    u_probex = torch.grid_sampler(ux.double(), probe_xy.double(), 0, 0, False).squeeze()
    u_probey = torch.grid_sampler(uy.double(), probe_xy.double(), 0, 0, False).squeeze()

    #return probe_xy.squeeze()
    return torch.stack((u_probex, u_probey), 2)
