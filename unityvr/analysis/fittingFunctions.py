import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from unityvr.analysis.utilityFunctions import getTrajFigName
from unityvr.viz import viz

from scipy.special import i0
from scipy.optimize import curve_fit

##functions to fit data to distributions

#von mises probability density function
def vonmises_pdf(x, mu, kappa):
    #x, mu are in radians
    V = np.exp((kappa)*np.cos((x-mu)))/(2*np.pi*i0(kappa))
    return V

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
    
    V = vonmises_pdf(np.linspace(0,2*np.pi,num=50), params[0], params[1])

    plt.plot(np.linspace(0,360,num=50), V, 'k-')
    plt.xlabel(r"$\theta$")
    if plotsave:
        plt.savefig(getTrajFigName("fit_vonmises",saveDir,uvrDat.metadata))

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(params[0], params[1], 'ro', alpha=0.5);
    ax.set_yticks([0.5,1])
    ax.set_theta_zero_location("E")
    ax.set_xticks(np.pi/180 * np.arange(-180,  180,  45))
    ax.set_thetalim(-np.pi, np.pi);
    
    if plotsave:
        fig.savefig(getTrajFigName("mu_kappa",saveDir,uvrDat.metadata))

        #returns mu in degree
    return params[0]*180/np.pi, params[1]


