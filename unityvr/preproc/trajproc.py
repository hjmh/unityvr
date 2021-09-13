import numpy as np
from numpy_ext import rolling_apply
import pandas as pd
import scipy as sp
import scipy.signal

import matplotlib.pyplot as plt

from unityvr.preproc import logproc
from unityvr.viz import viz

from os.path import sep, exists, isfile, join
from os import makedirs, listdir

from scipy.special import i0
from scipy.optimize import curve_fit

##functions to process posDf dataframe

#obtain the position dataframe with derived quantities
def position(uvrDat, derive = True, knownRdm = None, rotation=None):
    #TODO: describe input arguments... move to analysis folder
    # add argument for date
    # rotation: angle (degrees) by which to rotated the trajectory
    posDf = uvrDat.posDf

    #angle correction
    if (np.datetime64(uvrDat.metadata['date'])<=np.datetime64('2021-09-08')) & ('angle_convention' not in uvrDat.metadata):
        print('correcting for Unity angle convention.')
        posDf['angle'] = (-posDf['angle'])%360
        uvrDat.metadata['angle_convention'] = "right-handed"

    #rotate
    if rotation is not None:
        posDf['x'], posDf['y'] = rotation_deg(posDf['x'],posDf['y'],rotation)
        posDf['dx'], posDf['dy'] = rotation_deg(posDf['dx'],posDf['dy'],rotation)
        posDf['dxattempt'], posDf['dyattempt'] = rotation_deg(posDf['dxattempt'],posDf['dyattempt'],rotation)
        posDf['angle'] = (posDf['angle']+rotation)%360
        uvrDat.metadata['rotated_by'] = rotation

    #add radius metadata
    if knownRdm is not None:
        posDf.dc2cm = 10*knownRdm/uvrDat.metadata['ballRad']
    else:
        posDf.dc2cm = 10

    if derive:
        posDf['ds'] = np.sqrt(posDf['dx']**2+posDf['y']**2)
        posDf['s'] = np.cumsum(posDf['ds'])
        posDf['dTh'] = (np.diff(posDf['angle'],prepend=posDf['angle'].iloc[0]) + 180)%360 - 180
        posDf['radangle'] = ((posDf['angle']+180)%360-180)*np.pi/180

    return posDf

#segment flight bouts
def flightSeg(posDf, thresh, freq=120, plot = False, plotsave=False, saveDir=None, uvrDat=None, status='segmented'):

    #if masking, DO NOT UPDATE posDF in saved directory

    df = posDf.copy()

    _, t, F = sp.signal.spectrogram(df['ds'], freq)

    # first row of spectrogram seems to contain sufficient frequency information
    flight = sp.interpolate.interp1d(t,F[1,:]>thresh, kind='nearest', bounds_error=False)
    df['flight'] = flight(df['time'])

    if plot:
        plt.figure()
        plt.plot(t,F[1,:],'k');
        plt.plot(df['time'],df['flight']*F[1,:].max(),'r',alpha=0.2);
        plt.xlabel("time"); plt.legend(["power in HF band","thresholded"])

        fig, axs = plt.subplots(1,2,figsize=(10,5), gridspec_kw={'width_ratios':[20,1]})
        axs[0].plot(df['x']*posDf.dc2cm,df['y']*posDf.dc2cm,'-',linewidth=2,c='grey',alpha=0.1);
        cb = axs[0].scatter(df['x'].loc[df['flight']==0]*posDf.dc2cm,df['y'].loc[df['flight']==0]*posDf.dc2cm,
                                s=5,c=df['angle'].loc[df['flight']==0], cmap='twilight_shifted')
        axs[0].plot(df.x[0]*posDf.dc2cm,df.y[0]*posDf.dc2cm,'ok')
        axs[0].text(df.x[0]*posDf.dc2cm+0.2,df.y[0]*posDf.dc2cm+0.2,'start')
        axs[0].plot(df.x.values[-1]*posDf.dc2cm,df.y.values[-2]*posDf.dc2cm,'sk')
        axs[0].set_aspect('equal')
        axs[0].set_xlabel('x [cm]')
        axs[0].set_ylabel('y [cm]')

        viz.myAxisTheme(axs[0])
        plt.colorbar(cb,cax=axs[1], label='head direction [degree]');
        if plotsave:
            fig.savefig(saveDir+sep+'_'.join(['walking_trajectory_'+status, uvrDat.metadata['genotype'],
                                      uvrDat.metadata['sex'],
                                      uvrDat.metadata['flyid'],
                                      uvrDat.metadata['expid'][-5:],
                                      uvrDat.metadata['trial']+'.pdf']))

    df = carryAttrs(df,posDf)

    return df

#clip the dataframe
def flightClip(posDf, minT = 0, maxT = 485, plot = False, plotsave=False, saveDir=None, uvrDat=None, status='clipped'):

    #if masking, DO NOT UPDATE posDF in saved directory

    df = posDf.copy()

    df['clipped'] = ((posDf['time']<=minT) | (posDf['time']>=maxT))

    if plot:
        fig, axs = plt.subplots(1,2,figsize=(10,5), gridspec_kw={'width_ratios':[20,1]})
        axs[0].plot(df['x']*posDf.dc2cm,df['y']*posDf.dc2cm,'-',linewidth=2,c='grey',alpha=0.1);
        cb = axs[0].scatter(df['x'].loc[(df['clipped']==0)]*posDf.dc2cm,df['y'].loc[(df['clipped']==0)]*posDf.dc2cm,
                                s=5,c=df['angle'].loc[(df['clipped']==0)], cmap='twilight_shifted')
        axs[0].plot(df.loc[df['clipped']==0].x.iloc[0]*posDf.dc2cm,df.loc[(df['clipped']==0)].y.iloc[0]*posDf.dc2cm,'ok')
        axs[0].text(df.loc[(df['clipped']==0)].x.iloc[0]*posDf.dc2cm+0.2,df.loc[(df['clipped']==0)].y.iloc[0]*posDf.dc2cm+0.2,'start')
        axs[0].plot(df.loc[(df['clipped']==0)].x.iloc[-1]*posDf.dc2cm,df.loc[(df['clipped']==0)].y.iloc[-1]*posDf.dc2cm,'sk')
        axs[0].set_aspect('equal')
        axs[0].set_xlabel('x [cm]')
        axs[0].set_ylabel('y [cm]')
        viz.myAxisTheme(axs[0])
        plt.colorbar(cb,cax=axs[1], label='head direction [degree]');
        if plotsave:
            fig.savefig(saveDir+sep+'_'.join(['walking_trajectory_'+status, uvrDat.metadata['genotype'],
                                      uvrDat.metadata['sex'],
                                      uvrDat.metadata['flyid'],
                                      uvrDat.metadata['expid'][-5:],
                                      uvrDat.metadata['trial']+'.pdf']))

    df = carryAttrs(df,posDf)

    return df

#convert to shape space
def shape(posDf, plot = False, step = None, plotsave=False, saveDir=None, uvrDat=None, interp='linear',stitch=False):

    if 'clipped' in posDf: posDf = carryAttrs(posDf.loc[posDf['clipped']==0], posDf)
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


    if step is None:
        step = np.nanmedian(posDf['ds'])

    shapeDf = pd.DataFrame(columns = ['time','x','y','s','angle'])
    t_trans = sp.interpolate.interp1d(posDf['s'], posDf['time'], kind = interp)
    x_trans = sp.interpolate.interp1d(posDf['time'], posDf['x'], kind = interp)
    y_trans = sp.interpolate.interp1d(posDf['time'], posDf['y'], kind = interp)
    angle_trans = sp.interpolate.interp1d(posDf['time'], posDf['angle'], kind = interp)

    shapeDf['s'] = np.arange(np.nanmin(posDf['s']), np.nanmax(posDf['s']), step)

    shapeDf['time'] = t_trans(shapeDf['s'])
    shapeDf['x'] = x_trans(shapeDf['time'])
    shapeDf['y'] = y_trans(shapeDf['time'])
    shapeDf['angle'] = angle_trans(shapeDf['time'])

    shapeDf['dx'] = np.diff(shapeDf['x'], prepend = 0)
    shapeDf['dy'] = np.diff(shapeDf['y'], prepend = 0)

    shapeDf['ds'] = np.sqrt((shapeDf['dx']**2) + (shapeDf['dy']**2))

    shapeDf = carryAttrs(shapeDf,posDf)

    if plot:
        fig0 = plt.figure()
        ax1 = fig0.add_subplot(111)
        ax2 = ax1.twiny()
        ax1.plot(posDf['s'], posDf['time'], 'k');
        ax2.plot(np.cumsum(shapeDf['ds']), shapeDf['time'], 'r');
        ax1.set_ylabel("time")
        ax1.set_xlabel("$S_{time}$")
        ax2.set_xlabel("$S_{shape}$")

        fig, axs = plt.subplots(1,2,figsize=(10,5), gridspec_kw={'width_ratios':[20,1]})
        axs[0].plot(shapeDf['x']*shapeDf.dc2cm,shapeDf['y']*shapeDf.dc2cm,'-',linewidth=2,c='grey',alpha=0.1);
        cb = axs[0].scatter(shapeDf['x']*shapeDf.dc2cm,
                            shapeDf['y']*shapeDf.dc2cm,
                            s=5,c=shapeDf['angle'], cmap='twilight_shifted')
        axs[0].plot(shapeDf.x.iloc[0]*shapeDf.dc2cm,shapeDf.y.iloc[0]*shapeDf.dc2cm,'ok')
        axs[0].text(shapeDf.x.iloc[0]*shapeDf.dc2cm+0.2,shapeDf.y.iloc[0]*shapeDf.dc2cm+0.2,'start')
        axs[0].plot(shapeDf.x.values[-1]*shapeDf.dc2cm,shapeDf.y.values[-2]*shapeDf.dc2cm,'sk')
        axs[0].set_aspect('equal')
        axs[0].set_xlabel('x [cm]')
        axs[0].set_ylabel('y [cm]')
        viz.myAxisTheme(axs[0])
        plt.colorbar(cb,cax=axs[1], label='head direction [degree]');
        if plotsave:
            fig0.savefig(saveDir+sep+'_'.join(['shape_transformed_pathlength', uvrDat.metadata['genotype'],
                                      uvrDat.metadata['sex'],
                                      uvrDat.metadata['flyid'],
                                      uvrDat.metadata['expid'][-5:],
                                      uvrDat.metadata['trial']+'.pdf']))

    return shapeDf

#get pathlength
def pathC(ds):
    ds = np.array(ds)
    C = np.sum(ds)
    return C

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

    if plot:
        plt.plot(df['time'], df['tortuosity']);
        plt.xlabel("time"); plt.ylabel("C/L")
        plt.close()

        fig, axs = plt.subplots(1,2,figsize=(10,5), gridspec_kw={'width_ratios':[20,1]})
        axs[0].plot(df['x']*shapeDf.dc2cm,df['y']*shapeDf.dc2cm,'-',linewidth=2,c='grey',alpha=0.1);
        cb = axs[0].scatter(df['x']*shapeDf.dc2cm,
                            df['y']*shapeDf.dc2cm,
                            s=5,c=np.log(df['tortuosity']), cmap='viridis_r')
        axs[0].plot(df.x.iloc[0]*shapeDf.dc2cm,df.y.iloc[0]*shapeDf.dc2cm,'ok')
        axs[0].text(df.x.iloc[0]*shapeDf.dc2cm+0.2,df.y.iloc[0]*shapeDf.dc2cm+0.2,'start')
        axs[0].plot(df.x.values[-1]*shapeDf.dc2cm,df.y.values[-2]*shapeDf.dc2cm,'sk')
        axs[0].set_aspect('equal')
        axs[0].set_xlabel('x [cm]')
        axs[0].set_ylabel('y [cm]')
        viz.myAxisTheme(axs[0])
        plt.colorbar(cb,cax=axs[1], label='log(tortuosity)');
        if plotsave:
            fig.savefig(saveDir+sep+'_'.join(['local_tortuosity', uvrDat.metadata['genotype'],
                                      uvrDat.metadata['sex'],
                                      uvrDat.metadata['flyid'],
                                      uvrDat.metadata['expid'][-5:],
                                      uvrDat.metadata['trial']+'.pdf']))

    df = carryAttrs(df,shapeDf)

    return df


def vonmises_pdf(x, mu, kappa):
    #x, mu are in radians
    V = np.exp((kappa)*np.cos((x-mu)))/(2*np.pi*i0(kappa))
    return V

def fit_vonmises(degAngles, plot = False, binwidth = 20, plotsave=False, saveDir=None, uvrDat=None):
    # in degrees

    #width to radians
    binwidth = binwidth*np.pi/180

    #number of bins
    numbins = int(2*np.pi/binwidth)

    #convert to radians
    angles = degAngles*np.pi/180

    #get probability density and theta vector
    theta = np.linspace(0,2*np.pi,num=numbins+1)[:-1] + binwidth/2
    p = np.histogram(angles,bins=numbins,density=True)[0]

    plt.figure(figsize = (9,2))
    plt.step(theta*180/np.pi, p)

    #fit p as a function of theta
    params, _ = curve_fit(vonmises_pdf, theta, p, bounds=([0,0],[2*np.pi,np.inf]))

    #reconstruct fitted von mises distribution
    V = vonmises_pdf(np.linspace(0,2*np.pi,num=50), params[0], params[1])

    plt.plot(np.linspace(0,360,num=50), V, 'k-')
    plt.xlabel(r"$\theta$")
    if plotsave:
        plt.savefig(saveDir+sep+'_'.join(['fit_vonmises', uvrDat.metadata['genotype'],
                                    uvrDat.metadata['sex'],
                                    uvrDat.metadata['flyid'],
                                    uvrDat.metadata['expid'][-5:],
                                    uvrDat.metadata['trial']+'.pdf']))


    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(params[0], params[1], 'ro', alpha=0.5);
    ax.set_yticks([0.5,1])
    ax.set_theta_zero_location("E")
    ax.set_xticks(np.pi/180 * np.arange(-180,  180,  45))
    ax.set_thetalim(-np.pi, np.pi);
    if plotsave:
        fig.savefig(saveDir+sep+'_'.join(['mu_kappa', uvrDat.metadata['genotype'],
                                    uvrDat.metadata['sex'],
                                    uvrDat.metadata['flyid'],
                                    uvrDat.metadata['expid'][-5:],
                                    uvrDat.metadata['trial']+'.pdf']))

    return params[0]*180/np.pi, params[1]

def carryAttrs(df, posDf):
    attrs = list(posDf.__dict__)[5:]
    for a in attrs:
        df.__dict__[a] = posDf.__dict__[a]

    return df

def rotation_mat_rad(theta):
    #rotation matrix for an Nx2 vector with [x,y]
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return R

def rotation_deg(x,y,theta):
    theta = np.pi/180*theta
    r = np.matmul(np.array([x,y]).T,rotation_mat_rad(theta))
    return r[:,0], r[:,1]

#add the derived quantities and clipping information to the saved dataframe
def posDfUpdate(posDf, uvrDat, saveDir, saveName):

    #update uvrDat
    uvrDat.posDf = posDf
    savepath = uvrDat.saveData(saveDir, saveName,
                                     imaging=False)
    print("location:", saveDir)
