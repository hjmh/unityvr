### This module contains functions for plotting VR data, including functions to vizualize trajectories, frame rate, ...

import matplotlib.patches as mpatches
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np

def plotVRpathWithObjects(uvrExperiment,limx,limy, myfigsize):

    fig, ax = plt.subplots(1,1, figsize=myfigsize)
    
    ax = plotAllObjects(uvrExperiment, ax)

    ax.plot(uvrExperiment.posDf['x'], uvrExperiment.posDf['y'],color='grey',alpha=0.5)
    ax.scatter(uvrExperiment.posDf['x'], uvrExperiment.posDf['y'],s=7,c=uvrExperiment.posDf['time'],cmap='viridis')
    
    if np.isfinite(limx[0]):
        ax.set_ylim(limx[0], limx[1])
        ax.set_xlim(limy[0], limy[1])
    ax.set_aspect('equal')
    
    return fig

def plotAllObjects(uvrExperiment, ax):
    
    for obj in range(uvrExperiment.objDf.shape[0]):
        if("FlyCamera" not in uvrExperiment.objDf.name[obj]):
            #print(objDf['name'][obj])
            ellipse = mpatches.Ellipse((uvrExperiment.objDf['px'][obj], 
                                        uvrExperiment.objDf['py'][obj]), 
                                        uvrExperiment.objDf['sx'][obj], 
                                        uvrExperiment.objDf['sy'][obj], color='grey', alpha=0.5)
            ax.add_patch(ellipse)
            ax.annotate(uvrExperiment.objDf['name'][obj], (uvrExperiment.objDf['px'][obj]+5, uvrExperiment.objDf['py'][obj]-10))
    return ax