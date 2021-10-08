import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ppatch

# Functions related to characterizing bump position .......................................
## Circular calculation utilities
def circDistAbs(angle1,angle2):
    #angles should both be in radians
    return np.pi - abs(np.pi - abs(angle1-angle2))

def circDist(angle1,angle2):
    #angles should both be in radians and equal length

    if type(angle1) != np.ndarray:
        dist = (angle1-angle2)%(np.pi*2)
        if dist>np.pi: dist = dist - 2*np.pi
    else:
        dist = (np.unwrap(angle1)-np.unwrap(angle2))%(np.pi*2)
        dist[dist>np.pi] = dist[dist>np.pi] - 2*np.pi
    return dist


## Description of the (EB) bump related functions
def getRoiNum(df, roiname = 'slice'):
    roinames = [key for key in df.keys() if roiname in key ]
    return len(roinames)


def computePVA(locs, weights):
    """ Compute population vector average
    """
    nsteps = weights.shape[0]
    nvol = weights.shape[1]
    pva_x = np.cos(np.reshape(np.tile(locs,nvol),[nvol,nsteps])).T*weights
    pva_y = np.sin(np.reshape(np.tile(locs,nvol),[nvol,nsteps])).T*weights

    pva = np.vstack((sum(pva_x)/len(pva_x), sum(pva_y)/len(pva_x)))

    return pva


def getEBBumpPVA(df, roiname = 'slice'):
    roinames = [key for key in df.keys() if roiname in key ]
    nroi = len(roinames)

    roiArcPos = np.linspace(0, 2*np.pi, nroi+1)[:-1]
    roidat = df[roinames].values.T

    pva = computePVA(roiArcPos,roidat)

    pvaRad = np.mod(np.arctan2(pva[1,:],pva[0,:]), 2*np.pi)
    pvaLen = np.hypot(pva[0,:],pva[1,:])

    # flip to account for conversion to right-handed reference frame
    pvaRad = np.pi*2 - pvaRad

    return pvaRad, pvaLen, roiArcPos


# get max bump
def getMaxBumpPos(df, roiname='slice', order=3, window=7):
    from scipy.signal import savgol_filter
    roinames = [key for key in df.keys() if roiname in key ]

    roidat = df[roinames].values

    maxbump = savgol_filter(np.argmax(roidat,axis=1), window, order)

    # flip to account for conversion to right-handed reference frame
    return len(roinames)-maxbump


def shiftPVA(pva,offset):
    return (np.unwrap(pva) + offset)%(np.pi*2)


# Calcium traces vizualization .................................................
# Some ROI visualizations .......................................

def plotDFFheatmap(ax, df, roiname='slice', addColorbar=True,lefthanded=False):
    """
    Plot heatmap-style visualization of calcium imaging roi time series.
    We assume that calcium imaging rois are sorted in a left-handed rotational reference frame
    and flip the order to match the unity VR convention.
    """
    roinames = [key for key in df.keys() if roiname in key ]
    nroi = getRoiNum(df, roiname)

    order = np.arange(len(roinames),0,-1).astype('int')-1
    if lefthanded: order = np.arange(0,len(roinames)+1).astype('int')

    cax = ax.pcolor(df.posTime,order,df[roinames].values.T,cmap='Blues', edgecolors='face',shading='auto')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('\nROIs (n = {0})'.format(df[roinames].values.shape[1]))

    ax.set_ylim(-0.5,nroi-0.5)

    if addColorbar:
        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        cbar = fig.colorbar(cax)
        cbar.set_label('$(F - F_0) / F_0$ (per ROI)')  # vertically oriented colorbar

    return ax, cax
