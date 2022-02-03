### This module contains functions for plotting VR data, including functions to vizualize trajectories, frame rate, ...

import matplotlib.patches as mpatches
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
from os.path import sep, isfile, exists

from unityvr.viz import utils

## Velocity distributions
def plotVeloDistibution(ax,velo, nBins, binRange, xlim, xlabel,lineColor='dimgrey'):
    hist_velo,mybins = np.histogram(velo,bins=nBins, range=binRange,density=True)
    ax.plot(mybins[:-1]+0.5*np.diff(mybins), hist_velo,color=lineColor)
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    return ax


## Offset visualization
def summaryOffsetDetection(axs, offsetreg,rawoffset,kdevals,samplpts, kdepeaks, kdeOffsets, offsetcols, nroi):
    #plot histogram of raw offsets, pva-based offsets and KDE
    nbins=nroi*10
    axs= plotVeloDistibution(axs,offsetreg, nbins, (-np.pi, np.pi), (-np.pi, np.pi), '',lineColor='dimgrey')
    axs= plotVeloDistibution(axs,np.asarray([item for sublist in rawoffset for item in sublist]), nbins, (-np.pi, np.pi), (-np.pi, np.pi), \
                                 'Offset angle',lineColor='c')

    axs.plot(samplpts,kdevals*3,linewidth=3,color='teal',label='KDE')
    axs.set_xlim(-np.pi, np.pi)

    # plot peaks in color acording to label
    for l, p in enumerate(kdepeaks): axs.plot(kdeOffsets[l], kdevals[kdepeaks[l]]*3, "o",color=offsetcols[l],markersize=10)
    axs.legend(['PVA offset','DFF peak offsets','KDE'])
    utils.myAxisTheme(axs)
    return axs


## Fly paths
def plotFlyPath(uvrTest, convfac, figsize):
    fig, axs = plt.subplots(1,2,figsize=figsize, gridspec_kw={'width_ratios':[20,1]})
    axs[0].plot(uvrTest.posDf.x*convfac,uvrTest.posDf.y*convfac,color='grey', linewidth=0.5)
    cb = axs[0].scatter(uvrTest.posDf.x*convfac,uvrTest.posDf.y*convfac,s=5,c=uvrTest.posDf.angle, cmap='hsv')
    axs[0].plot(uvrTest.posDf.x[0]*convfac,uvrTest.posDf.y[0]*convfac,'ok')
    axs[0].text(uvrTest.posDf.x[0]*convfac+0.2,uvrTest.posDf.y[0]*convfac+0.2,'start')
    axs[0].plot(uvrTest.posDf.x.values[-1]*convfac,uvrTest.posDf.y.values[-2]*convfac,'sk')
    plt.colorbar(cb,cax=axs[1], label='head direction [degree]')

    return fig, axs

def plotVRpathWithObjects(uvrExperiment,limx,limy, myfigsize):

    fig, ax = plt.subplots(1,1, figsize=myfigsize)

    ax = plotAllObjects(uvrExperiment, ax)

    ax.plot(uvrExperiment.posDf['x'], uvrExperiment.posDf['y'],color='grey',alpha=0.5)
    ax.scatter(uvrExperiment.posDf['x'], uvrExperiment.posDf['y'],s=7,c=uvrExperiment.posDf['time'],cmap='viridis')

    if np.isfinite(limx[0]):
        ax.set_xlim(limx[0], limx[1])
        ax.set_ylim(limy[0], limy[1])
    ax.set_aspect('equal')

    return fig, ax

def plotAllObjects(uvrExperiment, ax):

    for obj in range(uvrExperiment.objDf.shape[0]):
        if("Plane" in uvrExperiment.objDf.name[obj]): continue

        if("FlyCamera" not in uvrExperiment.objDf.name[obj]):
            ax = plotObjectEllipse(ax,
                                   [uvrExperiment.objDf['sx'][obj], uvrExperiment.objDf['sy'][obj]],
                                   [uvrExperiment.objDf['px'][obj], uvrExperiment.objDf['py'][obj]])
            ax.annotate(uvrExperiment.objDf['name'][obj], (uvrExperiment.objDf['px'][obj]+5, uvrExperiment.objDf['py'][obj]-10))
    return ax


def plotObjectEllipse(ax, rad, pos):
    ellipse = mpatches.Ellipse((pos[0],pos[1]), rad[0], rad[1], color='grey', alpha=0.5)
    ax.add_patch(ellipse)

    return ax


def plotTraj(ax,xpos,ypos,param,size=5,unit="cm", cmap='twilight_shifted',limvals=(0,360)):

    cb = ax.scatter(xpos,ypos,s=size,c=param,cmap=cmap, vmin=limvals[0], vmax=limvals[1])
    ax.plot(xpos[0],ypos[0],'ok')
    ax.text(xpos[0]+0.2,ypos[0]+0.2,'start')
    ax.plot(xpos[-1],ypos[-1],'sk')

    ax.set_aspect('equal')
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    utils.myAxisTheme(ax)

    return ax, cb


def plotTrajwithParameterandCondition(df, figsize, parameter='angle',
                                      condition=None,
                                      color = 'grey',
                                      mycmap = 'twilight_shifted',
                                      transform = lambda x: x,
                                      plotOriginal=True,
                                      mylimvals = (0,360)
                                     ):

    if condition is None: condition = np.ones(np.shape(df['x']),dtype='bool')

    fig, axs = plt.subplots(1,2,figsize=figsize, gridspec_kw={'width_ratios':[20,1]})

    if plotOriginal:
        axs[0].plot(df['x']*df.dc2cm,df['y']*df.dc2cm,color=color, linewidth=0.5)

    axs[0],cb = plotTraj(axs[0],df.loc[condition].x.values*df.dc2cm,
                         df.loc[condition].y.values*df.dc2cm,
                         df[parameter].loc[condition].transform(transform),
                         5,"cm", mycmap, mylimvals)

    plt.colorbar(cb,cax=axs[1],label=parameter)

    return fig, axs
