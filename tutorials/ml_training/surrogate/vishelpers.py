import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

    
def pcolor_list(list_of_images, title=""):
    try:
        fig, axes = plt.subplots(1, len(list_of_images))
        
        fig.set_size_inches(4*len(list_of_images),4)
        for (ax, mat) in zip(axes, list_of_images):
            ax.pcolor(mat, cmap=cm.coolwarm, clim=[0,1])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
    except:
        fig = plt.figure(figsize=(4,4))
        plt.pcolor(list_of_images)
        plt.axis("off", cmap=cm.coolwarm, clim=[0,1])
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        
    plt.suptitle(title, y=0.92, fontsize=14)
    plt.show()

    fig = plt.figure(figsize=(0.00001, 0.00001))
    dummy = plt.plot(0,0)
    gca = plt.gca()
    gca.set_visible(False)
    plt.show(dummy)