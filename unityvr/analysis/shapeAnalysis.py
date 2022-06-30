import numpy as np
from numpy_ext import rolling_apply
import pandas as pd
import scipy as sp

from skimage.filters import threshold_otsu
from scipy.stats import skew, kurtosis

from os.path import sep, exists, join

import matplotlib.pyplot as plt

from unityvr.viz import viz
from unityvr.analysis.utils import carryAttrs
from unityvr.analysis.utils import getTrajFigName

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
            
            counter = 'count' if 'count' in posDf else 'frame'
            
            for i,fstart in enumerate(np.array(posDf[counter].loc[posDf['flight'].diff()==1])):
                fstop = np.array(posDf[counter].loc[posDf['flight'].diff()==-1])[i]
                df.loc[df[counter]>=fstop,'x'] += float(posDf.loc[posDf[counter]==fstart-1]['x'])-float(posDf.loc[posDf[counter]==fstop]['x'])
                df.loc[df[counter]>=fstop,'y'] += float(posDf.loc[posDf[counter]==fstart-1]['y'])-float(posDf.loc[posDf[counter]==fstop]['y'])
            posDf = carryAttrs(df.dropna(subset=['x','y']),posDf)

    # if the step length is not specified, choose the mean velocity of the fly
    if step is None:
        step = np.nanmean(posDf['ds'])

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
        ax01.plot(posDf['s'], 'k', label = r"$S_{time}$");
        ax02.plot(shapeDf['s'], 'r', label = r"$S_{shape}$");
        ax01.set_ylabel(r"$S$")
        fig0.legend(loc = "center right")

        fig1, ax1 = viz.plotTrajwithParameterandCondition(shapeDf, figsize=(10,5), parameter='angle')
        if plotsave:
            fig1.savefig(getTrajFigName("walking_trajectory_shape_space",saveDir,uvrDat.metadata))
            fig0.savefig(getTrajFigName("transformed_cumulative_pathlength",
                                    saveDir,uvrDat.metadata))

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
def tortuosityLoc(shapeDf, window=None, window_cm = 5 #in cm
                  , plot = False, plotsave=False, saveDir=None, uvrDat=None):
    
    df = shapeDf.copy()
    
    #decimeter value overrides cm values
    if window is None: window = int((window_cm/shapeDf.dc2cm)/(np.median(shapeDf['ds'])))
    
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
        plt.figure()
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

def maximize_bim_coeff(shapeDf, lims = (100,5000), res = 1.5, plot = False):

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

def extractVoltes(shapeDf, res_cm = 0.5, L_thresh_min_cm = 1 #in cm
                  , L_thresh_max_cm = 10 #in cm
                  , res = None, L_thresh_min = None, L_thresh_max = None
                  , plot = False, plotsave=False, saveDir=None, uvrDat=None):
    
    #decimeter values override cm values
    if res is None: res = res_cm/shapeDf.dc2cm
    if L_thresh_min is None: L_thresh_min = L_thresh_min_cm/shapeDf.dc2cm
    if L_thresh_max is None: L_thresh_max = L_thresh_max_cm/shapeDf.dc2cm
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
    
    if plot:
        fig, ax = viz.plotTrajwithParameterandCondition(df, figsize=(10,5),
                                        condition=df['voltes'])
        if plotsave:
            fig.savefig(getTrajFigName("walking_trajectory_voltes",saveDir,uvrDat.metadata))

    return df

def shapeToTime(posDf,shapeDf,label,new_name=None):
    
    data_type = shapeDf.dtypes[label]
    if data_type == 'bool':
        interp_type = 'int'
        interp_kind = 'nearest'
        #fill = 0
    else:
        interp_type = data_type
        interp_kind = 'linear'
        #fill = float("NaN")
    
    pDf = posDf.copy()

    transform = sp.interpolate.interp1d(shapeDf['time'],shapeDf[label].astype(interp_type),kind=interp_kind,bounds_error=False,fill_value="extrapolate")
    
    if new_name is None:
        pDf[label] = transform(pDf['time']).astype(data_type)
    else:
        pDf[new_name] = transform(pDf['time']).astype(data_type)

    pDf = carryAttrs(pDf, posDf)

    return pDf

def number_of_voltes(shapeDf):
    return sp.ndimage.label(shapeDf['voltes'])[1]

def volte_tortuosity_difference(shapeDf):
    return np.nanmean(np.log(shapeDf.loc[shapeDf['voltes']]['tortuosity']))-np.nanmean(np.log(shapeDf.loc[~shapeDf['voltes']]['tortuosity']))
