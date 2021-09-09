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

# axis beautification
def myAxisTheme(myax):
    myax.get_xaxis().tick_bottom()
    myax.get_yaxis().tick_left()
    myax.spines['top'].set_visible(False)
    myax.spines['right'].set_visible(False)

def plotScaleBar(ax,xlen,pos,labeltext):
    ax.plot([pos[0],pos[0]+xlen],[pos[1],pos[1]],'k')
    ax.text(pos[0],pos[1],labeltext)

def minimalAxisTheme(myax, xlen,pos,labeltext):
    plotScaleBar(myax,xlen,pos,labeltext)
    myax.axis('off')
    myax.set_aspect('equal')

def pathPlotAxisTheme(myax, units):
    myax.spines['top'].set_visible(False)
    myax.spines['right'].set_visible(False)
    myax.spines['bottom'].set_visible(False)
    myax.spines['left'].set_visible(False)
    myax.get_xaxis().set_ticks([])
    myax.get_yaxis().set_ticks([])
    myax.set_aspect('equal')
    myax.set_xlabel('x [{}]'.format(units))
    myax.set_ylabel('y [{}]'.format(units))
