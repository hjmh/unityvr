import matplotlib.pyplot as plt
from os.path import sep, exists
from os import makedirs


## Utilities

def addlabs(axs, xlabs, ylabs):
    for i,ax in enumerate(axs):
        ax.set_xlabel(xlabs[i])
        ax.set_ylabel(ylabs[i])
        
def addlims(axs, xlims, ylims):
    for i,ax in enumerate(axs):
        ax.set_xlim(xlabs[i])
        ax.set_ylim(ylabs[i])
        
def makemydir(myDir):
    if not exists(myDir): makedirs(myDir)