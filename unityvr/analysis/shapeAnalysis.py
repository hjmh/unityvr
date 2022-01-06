import numpy as np
from numpy_ext import rolling_apply
import pandas as pd
import scipy as sp

from skimage.filters import threshold_otsu
from scipy.stats import skew, kurtosis

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
            posDf = carryAttrs(df.dropna(),posDf)

    # if the step length is not specified, choose the median velocity as the step length
    if step is None:
        step = np.nanmedian(posDf['ds'])
    
    # get trajectory
    points = np.array([posDf['x'].values,posDf['y'].values]).T
    _,idx = np.unique(points,axis=0,return_index=True) #remove repeats
    clean = np.array([points[i] for i in sorted(idx)])
    time = np.array([posDf['time'].iloc[i] for i in sorted(idx)])
    angle = np.array([posDf['angle'].iloc[i] for i in sorted(idx)])

    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(clean, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]
    
    # Interpolation for different methods:
    alpha = np.linspace(0, 1, int(np.nanmax(posDf['s'])/step))

    path_interpolator =  sp.interpolate.interp1d(distance, clean, kind=interp, axis=0)
    time_interpolator = sp.interpolate.interp1d(distance, time, kind=interp,axis=0)
    angle_interpolator = sp.interpolate.interp1d(distance, angle, kind=interp,axis=0)

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

def bimodality_coeff(shapeDf):
    gam = skew(shapeDf['tortuosity'].transform(lambda x: np.log(x)).dropna())
    
    kap = kurtosis(shapeDf['tortuosity'].transform(lambda x: np.log(x)).dropna())
    
    n = len(shapeDf['tortuosity'].transform(lambda x: np.log(x)).dropna())
    
    b = ((gam**2) + 1)/(kap + 3*((n-1)**2)/((n-2)*(n-3)))
    
    return b

def maximize_bim_coeff(shapeDf, lims = (10,1000), res = 1, plot = False):
    
    windows = np.round(np.exp(np.arange(np.log(lims[0]),np.log(lims[1]),res))).astype('int')
    
    beta = np.zeros(np.shape(windows))
    
    for i,window in enumerate(windows):
        shapeDftemp = tortuosityLoc(shapeDf, window=window)
        beta[i] = bimodality_coeff(shapeDftemp)
        
    win_max = windows[beta==np.nanmax(beta)][-1]
    
    if plot:
        shapeDffin = tortuosityLoc(shapeDf, window=win_max)
        with pd.option_context('mode.use_inf_as_na', True):
            shapeDffin['tortuosity'].transform(lambda x: np.log(x)).dropna().plot.kde()
            
    return win_max

def intersection(x1,x2,x3,x4,y1,y2,y3,y4):
    #finds all the intersections points between 2 lines
    
    d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if d:
        xs = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / d
        ys = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / d
        if (xs >= min(x1,x2) and xs <= max(x1,x2) and
            xs >= min(x3,x4) and xs <= max(x3,x4)):
            return xs, ys
        
def extractVoltes(shapeDf, res=0.05, L_thresh_min = 0.1, L_thresh_max = 1):
    
    #resolution of x only considers points spaced x distance apart on the trajectory to find intersections
    
    df = shapeDf.copy()
    
    #convert path-length resolution to step-resolution
    step_res = np.where(df['s']>=res)[0][0]
    
    x = df['x'].iloc[::step_res].values
    y = df['y'].iloc[::step_res].values
    t = df['time'].iloc[::step_res].values

    xs, ys = [], []
    ts = []
    for i in range(len(x)-1):
        for j in range(i-1):
            if xs_ys := intersection(x[i],x[i+1],x[j],x[j+1],y[i],y[i+1],y[j],y[j+1]):
                xs.append(xs_ys[0])
                ys.append(xs_ys[1])
                ts.append([t[j],t[i]])

    ts = np.array(ts)
    
    con_net = np.zeros(np.shape(shapeDf['time'])).astype('bool')
    for i in range(len(ts)):
        con = (df['time']>=ts[i,0]) & (df['time']<=ts[i,1])
        L = np.sum(shapeDf['ds'][con])
        if (L>=L_thresh_min) & (L<=L_thresh_max):
            con_net = (con_net)|(con)
    
    df['voltes'] = con_net
    
    df = carryAttrs(df, shapeDf)
    
    return df

def shapeToTimeBoolean(posDf,shapeDf,label):
    
    pDf = posDf.copy()
    
    transform = sp.interpolate.interp1d(shapeDf['time'],shapeDf[label].astype('int'),kind="nearest",bounds_error=False,fill_value=0)
    
    pDf[label] = transform(pDf['time']).astype('bool')
    
    pDf = carryAttrs(pDf, posDf)
    
    return pDf

def number_of_voltes(shapeDf):
    return sp.ndimage.label(shapeDf['voltes'])[1]

def volte_tortuosity_difference(shapeDf):
    return np.nanmean(np.log(shapeDf.loc[shapeDf['voltes']]['tortuosity']))-np.nanmean(np.log(shapeDf.loc[~shapeDf['voltes']]['tortuosity']))

def shapeDfUpdate(shapeDf, uvrDat, saveDir, saveName):
    savepath = sep.join([saveDir,saveName,'uvr'])

    #update uvrDat
    shapeDf.to_csv(sep.join([savepath,'shapeDf.csv']))
    print("location:", saveDir)