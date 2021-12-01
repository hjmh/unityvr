import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from unityvr.analysis.utilityFunctions import getTrajFigName
from unityvr.viz import viz

from scipy.special import i0
from scipy.optimize import curve_fit
import scipy.stats as sts

##functions to fit data to distributions

#von mises probability density function
def vonmises_pdf(x, mu, kappa):
    #x, mu are in radians
    V = np.exp((kappa)*np.cos((x-mu)))/(2*np.pi*i0(kappa))
    return V

def sum_of_vonmises_pdf(x, mu1, kappa1, mu2, kappa2):
    V = np.exp((kappa1)*np.cos((x-mu1)))/(2*np.pi*i0(kappa1)) + np.exp((kappa2)*np.cos((x-mu2)))/(2*np.pi*i0(kappa2))

#function to fit data to the von mises pdf
def fit_vonmises(degAngles, binwidth = 20, plot = False, plotsave=False, saveDir=None, uvrDat=None):
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
    fit_func = vonmises_pdf(theta, params[0], params[1])
    
    #compute kolmogorov-smirnoff stat
    [ks, p_value] = sts.ks_2samp(p, fit_func)
    
    #compute squared difference from fit
    sqd = np.sum(np.square(p-fit_func))
    
    #decide whether distribution is unimodal
    notFit = np.logical_and(np.logical_and(p_value<=0.1, params[1]<0.3),~np.isnan(params[1]))
    
    #if not:
    #fit p as a function of theta to a sum of vonmises
    if notFit:
        params, _ = curve_fit(sum_of_vonmises_pdf, theta, p, bounds=([0,0,0,0],[2*np.pi,np.inf,2*np.pi,np.inf]))
        fit_func = sum_of_vonmises_pdf(theta, params[0], params[1], params[2], params[3])

        #compute kolmogorov-smirnoff stat
        [ks, p_value] = sts.ks_2samp(p, fit_func)

        #compute squared difference from fit
        sqd = np.sum(np.square(p-fit_func))
    
    if plot:
        if notFit: V = sum_of_vonmises_pdf(np.linspace(0,2*np.pi,num=50), params[0], params[1], params[2], params[3])
        else: V = vonmises_pdf(np.linspace(0,2*np.pi,num=50), params[0], params[1])
        
        #plot 1
        plt.plot(np.linspace(0,360,num=50), V, 'k-')
        plt.xlabel(r"$\theta$")
        
        if plotsave:
            plt.savefig(getTrajFigName("fit_vonmises",saveDir,uvrDat.metadata))
        
        #plot 2
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        if notFit: 
            ax.plot(params[0], params[1], 'ro', alpha=0.5)
            ax.plot(params[2], params[3], 'bo', alpha=0.5)
        else: ax.plot(params[0], params[1], 'ro', alpha=0.5)
        ax.set_yticks([0.5,1])
        ax.set_theta_zero_location("E")
        ax.set_xticks(np.pi/180 * np.arange(-180,  180,  45))
        ax.set_thetalim(-np.pi, np.pi);

        if plotsave:
            fig.savefig(getTrajFigName("mu_kappa",saveDir,uvrDat.metadata))

        #returns mu in degree
    
    mu1 = params[0]*180/np.pi
    kappa1 = params[1]
    mu2 = params[2]*180/np.pi if notFit else None
    kappa2 = params[3] if notFit else None
    
    return mu1, kappa1, mu2, kappa2, ks, p_value, sqd

