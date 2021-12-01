import numpy as np
from numpy_ext import rolling_apply
import pandas as pd
import scipy as sp

from skimage.filters import threshold_otsu

from os.path import sep, exists, join

import matplotlib.pyplot as plt

from unityvr.viz import viz
from unityvr.analysis.utilityFunctions import carryAttrs
from unityvr.analysis.utilityFunctions import getTrajFigName

##functions to derive and process shapeDf dataframe

#convert to shape space
def shape(posDf, step = None, interp='linear', stitch=False, plot = False, plotsave=False, saveDir=None, uvrDat=None):

    #if the posDf has been segmented and clipped to remove regions of flight
    if 'clipped' in posDf: 
        posDf = carryAttrs(posDf.loc[posDf['clipped']==0], posDf)
        pDf = posDf.copy(); pDf.loc[:,'x'] -= float(pDf['x'].iloc[0]); pDf.loc[:,'y'] -= float(pDf['y'].iloc[0])
        posDf = carryAttrs(pDf,posDf)
    if 'flight' in posDf:
        if not stitch:
            posDf = carryAttrs(posDf.loc[posDf['flight']==0], posDf)
            interp = 'nearest'
        if stitch:
            df = posDf.where(posDf['flight']==0).copy()
            for i,fstart in enumerate(np.array(posDf['frame'].loc[posDf['flight'].diff()==1])):
                fstop = np.array(posDf['frame'].loc[posDf['flight'].diff()==-1])[i]
                df.loc[df['frame']>=fstop,'x'] += float(posDf.loc[posDf['frame']==fstart-1]['x'])-float(posDf.loc[posDf['frame']==fstop]['x'])
                df.loc[df['frame']>=fstop,'y'] += float(posDf.loc[posDf['frame']==fstart-1]['y'])-float(posDf.loc[posDf['frame']==fstop]['y'])
            posDf = carryAttrs(df,posDf)

    # if the step length is not specified, choose the median velocity as the step length
    if step is None:
        step = np.nanmedian(posDf['ds'])
    
    # get trajectory
    points = np.array([posDf['x'].values,posDf['y'].values]).T
    _,idx = np.unique(points,axis=0,return_index=True) #remove repeats
    clean = np.array([points[i] for i in sorted(idx)])
    time = np.array([posDf.loc[i,'time'] for i in sorted(idx)])
    angle = np.array([posDf.loc[i,'angle'] for i in sorted(idx)])

    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(clean, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]

    # Interpolation for different methods:
    alpha = np.linspace(0, 1, int(posDf['s'].iloc[-1]/step))

    path_interpolator =  sp.interpolate.interp1d(distance, clean, kind='nearest', axis=0)
    time_interpolator = sp.interpolate.interp1d(distance, time, kind='nearest',axis=0)
    angle_interpolator = sp.interpolate.interp1d(distance, angle, kind='nearest',axis=0)

    shapeDf = pd.DataFrame(path_interpolator(alpha),columns = ['x','y'])
    shapeDf['time'] = time_interpolator(alpha)
    shapeDf['angle'] = angle_interpolator(alpha)
    shapeDf['dx'] = np.diff(shapeDf['x'],prepend=0)
    shapeDf['dy'] = np.diff(shapeDf['y'],prepend=0)
    shapeDf['ds'] = np.sqrt((shapeDf['dx']**2)+(shapeDf['dy']**2))
    shapeDf['s'] = np.cumsum(shapeDf['ds'])

    shapeDf = carryAttrs(shapeDf,posDf)
    
    if plot:
        fig0 = plt.figure()
        ax01 = fig0.add_subplot(111)
        ax02 = ax01.twiny()
        ax01.plot(posDf['s'], posDf['time'], 'k', label = r"$S_{time}$");
        ax02.plot(np.cumsum(shapeDf['ds']), shapeDf['time'], 'r', label = r"$S_{shape}$");
        ax01.set_ylabel("time")
        ax01.set_xlabel(r"$S_{time}$")
        ax02.set_xlabel(r"$S_{shape}$")
        fig0.legend(loc = "center right")
        
        fig1, ax1 = viz.plotTrajwithParameterandCondition(shapeDf, figsize=(10,5), parameter='angle')
        if plotsave:
            fig1.savefig(getTrajFigName("walking_trajectory_shape_space",saveDir,uvrDat.metadata))

    return shapeDf

#get pathlength
def pathC(ds):
    ds = np.array(ds)
    C = np.sum(ds)
    return C

#get shortest distance between path start and path end
def pathL(x,y):
    x = np.array(x); y = np.array(y)
    L = np.sqrt((x[-1]-
             x[0])**2 + ((y[-1]-y[0]))**2)
    return L

#get net tortuosity
def tortuosityGlo(x, y, ds):
    return pathC(ds)/pathL(x,y)

#get local tortuosity
def tortuosityLoc(shapeDf, window=500, plot = False, plotsave=False, saveDir=None, uvrDat=None):
    df = shapeDf.copy()
    df['tortuosity'] = rolling_apply(tortuosityGlo, window, df['x'], df['y'], df['ds'])

    df = carryAttrs(df,shapeDf)
    
    if plot:
        fig, ax = viz.plotTrajwithParameterandCondition(df, figsize=(10,5), 
                                        parameter='tortuosity', mycmap='viridis_r', mylimvals=[None, None], transform = lambda x: np.log(x))
        if plotsave:
            fig.savefig(getTrajFigName("walking_trajectory_tortuosity",saveDir,uvrDat.metadata))

    return df

def segment(shapeDf, plot=False):
    
    df = shapeDf.copy()
    
    thresh = threshold_otsu(df['tortuosity'].transform(lambda x: np.log(x)).dropna())
    df['curvy'] = np.log(df['tortuosity'])>thresh
    
    if plot:
        with pd.option_context('mode.use_inf_as_na', True):
            df['tortuosity'].transform(lambda x: np.log(x)).dropna().plot.kde()
        plt.axvline(thresh,color='k')
        
    df = carryAttrs(df,shapeDf)
        
    return df

def shapeDfUpdate(shapeDf, uvrDat, saveDir, saveName):
    savepath = sep.join([saveDir,saveName,'uvr'])

    #update uvrDat
    shapeDf.to_csv(sep.join([savepath,'shapeDf.csv']))
    print("location:", saveDir)
