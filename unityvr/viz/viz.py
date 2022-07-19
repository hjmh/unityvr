### This module contains functions for plotting VR data, including functions to vizualize trajectories, frame rate, ...

import matplotlib.patches as mpatches
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
from os.path import sep, isfile, exists
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns

from unityvr.viz import utils

# general purpose
def stripplotWithLines(df, valvar, groupvar,huevar, axs, xlab, ylab, ylimvals,
                       filtering=False, filterval=0, filtervar='None', normalize=False, normgroup='None', palette='tab20b_r', order='None'):
    import matplotlib.colors as colors

    samples = list(np.unique(df[huevar].values))
    groups = list(np.unique(df[groupvar].values))

    # generate color map
    cNorm  = colors.Normalize(vmin=1, vmax=len(samples))
    myCMap = plt.cm.ScalarMappable(norm=cNorm,cmap=palette)

    for s, sample in enumerate(samples):
        sampledf = df.query('{}=="{}"'.format(huevar,sample))

        toplot = sampledf[valvar].values
        groupnames = sampledf[groupvar].values

        #mask mean position where offset PVA length was low
        if filtering:
            filtervariable = sampledf[filtervar].values
            toplot[filtervariable>filterval] = np.nan

        #noramlize if desired
        if normalize:
            toplot = toplot - toplot[normgroup]
            if ylimvals == (-np.pi, np.pi):
                toplot[toplot>np.pi] = toplot[toplot>np.pi] - 2*np.pi
                toplot[toplot<-np.pi] = toplot[toplot<-np.pi] + 2*np.pi

        jitter = np.random.uniform(low=-0.1,high=0.1, size=len(groups))

        if order != 'None':
            groupnames = [groupnames[i] for i in order]
            toplot = [toplot[i] for i in order]
            axs.plot(np.arange(len(groups))+jitter, toplot,'o', color=myCMap.to_rgba(s+1), label=sample)
            axs.plot(np.arange(len(groups))+jitter, toplot,'-', color=myCMap.to_rgba(s+1))
        else:
            axs.plot(np.arange(len(groups))+jitter, toplot,'o', color=myCMap.to_rgba(s+1), label=sample)
            axs.plot(np.arange(len(groups))+jitter, toplot,'-', color=myCMap.to_rgba(s+1))


    # beautification
    axs.set_ylabel(ylab)
    axs.set_ylim(ylimvals)
    axs.set_xlabel(xlab)
    axs.set_xticks(np.arange(len(groups)))
    axs.set_xticklabels(groupnames)
    return axs


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

def plotAllObjects(uvrExperiment, ax, labelobj=False, objsize=(-1,-1)):

    for obj in range(uvrExperiment.objDf.shape[0]):
        if("Plane" in uvrExperiment.objDf.name[obj]): continue

        if("FlyCamera" not in uvrExperiment.objDf.name[obj]):
            if objsize[0] < 0:
                ax = plotObjectEllipse(ax,
                                   [uvrExperiment.objDf['sx'][obj], uvrExperiment.objDf['sy'][obj]],
                                   [uvrExperiment.objDf['px'][obj], uvrExperiment.objDf['py'][obj]])
            else:
                ax = plotObjectEllipse(ax, objsize,
                                      [uvrExperiment.objDf['px'][obj], uvrExperiment.objDf['py'][obj]])
            if labelobj:
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
    ax.set_xlabel('x [{}]'.format(unit))
    ax.set_ylabel('y [{}]'.format(unit))
    utils.myAxisTheme(ax)

    return ax, cb


def plotTrajwithParameterandCondition(df, figsize, parameter='angle',
                                      condition=None,
                                      color = 'grey',
                                      mycmap = 'twilight_shifted',
                                      transform = lambda x: x,
                                      plotOriginal=True,
                                      stitch = False,
                                      mylimvals = (0,360)
                                     ):

    if condition is None: condition = np.ones(np.shape(df['x']),dtype='bool')

    fig, axs = plt.subplots(1,2,figsize=figsize, gridspec_kw={'width_ratios':[20,1]})
    
    if stitch:
        x_label='x_stitch'
        y_label='y_stitch'
    else:
        x_label='x'
        y_label='y'
    
    if plotOriginal:
        axs[0].plot(df[x_label]*df.dc2cm,df[y_label]*df.dc2cm,color=color, linewidth=0.5)
    
    if len(df.loc[condition])>0:
        axs[0],cb = plotTraj(axs[0],df.loc[condition,x_label].values*df.dc2cm,
                             df.loc[condition,y_label].values*df.dc2cm,
                             df[parameter].loc[condition].transform(transform),
                             5,"cm", mycmap, mylimvals)
        plt.colorbar(cb,cax=axs[1],label=parameter)

    return fig, axs

def circ_point_dist_plotter(ax,degrees,bin_size,zero_direction,
                            start_ang,bottom,min_ax_ang,max_ax_ang,
                            rounds_res=1,sign=0,color='k',
                            alpha=0.85,markersize=5,
                            pointscale=0.15,
                            binscale=0.27,ylim_max=2):
    
    a, b = np.histogram(degrees, bins=np.arange(start_ang, 360+bin_size+start_ang, bin_size))
    centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])
    #ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=bottom, alpha = 0.5, edgecolor='gray')
    for i in range(np.max(a)):
        ax.plot(centers[(a-i)>0]-(np.deg2rad(bin_size*binscale)*sign),
                bottom+(a[(a-i)>0]-i)*pointscale,'o',color=color,alpha=alpha,markersize=markersize)
    ax.set_theta_zero_location(zero_direction)
    ax.set_thetamin(min_ax_ang)
    ax.set_thetamax(max_ax_ang)
    ax.set_ylim(0,ylim_max)
    
def full_circular_plotter(ax, df, cond, k_min, k_max, R, muvar='mu', color='grey', sign=0, alpha=0.85, label=None, outerpoints=True):
    
    x = df[muvar].where(cond & (df['kappa']>k_min)).astype('float').values
    pva = np.nanmean(np.exp(1j*np.deg2rad(x)))
    
    if outerpoints:
        circ_point_dist_plotter(ax,x,10,"N",0,1,-180,180,alpha=alpha,color=color,sign=sign)

    ##MODIFY HEAD LENGTH AND WIDTH IF ARROW NOT VISIBLE
    ax.arrow(np.angle(pva), 0, 0, np.round(np.abs(pva),2), width = 0.1, lw = 0.1,
        head_width = 0.23, head_length=0.11, color=color, label=label, zorder=1, length_includes_head=True)

    #max condition only applies to lines
    x_prime = df[muvar].where(cond & (df['kappa']<k_max)).astype('float').values*np.pi/180
    y = df['kappa'].where(cond & (df['kappa']<k_max)).astype('float').values/R

    for j,_ in enumerate(x):
        ax.plot([0,x_prime[j]],[0,y[j]],ls='-',alpha=alpha,zorder=-1,color=color)
        
def linear_ordered_plotter(ax, inDf, order, variable, colormapper = None, hue_var='flyid', ylim=(-190,190)):
    
    pal = "tab20b_r" if (colormapper is None) else sns.color_palette(colormapper)
    
    df = inDf.copy()
    
    stims = pd.DataFrame([a + b for a, b in zip(list(order)[::2],list(order)[1::2])],columns={'s'})
    stims['t'] = "t"+(stims.groupby('s').cumcount()+1).apply(lambda x: "{:02d}".format(x))
    stims = CategoricalDtype(list(stims['s']+stims['t']), ordered=True)

    df['stimtrial'] = df['stimtrial'].astype(stims)

    sns.lineplot(x = 'stimtrial', y = variable, hue=hue_var, palette=pal, data = 
                          df,  ax =ax);

    sns.stripplot(x = 'stimtrial', y = variable, hue=hue_var, palette=pal, data = 
                          df,  ax =ax, jitter=False, label=str());
    ax.set_ylim(ylim)
    
    N = len(df[hue_var].unique())
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:N],labels[:N],bbox_to_anchor=(1.2, 1.2))
    sns.despine()
    
