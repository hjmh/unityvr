# Functions for aligning imaging and VR data
import numpy as np
import matplotlib.pyplot as plt
from unityvr.viz import utils
import pandas as pd
from os.path import sep
import json
from unityvr.preproc import logproc
from unityvr.analysis import utils

def findImgFrameTimes(uvrDat,imgMetadat):

    imgInd = np.where(np.diff(uvrDat.nidDf.imgfsig.values)>3)[0]

    imgFrame = uvrDat.nidDf.frame.values[imgInd].astype('int')

    #take only every x frame as start of volume
    volFrame = imgFrame[0::imgMetadat['fpv']]
    volFramePos = np.where(np.in1d(uvrDat.posDf.frame.values,volFrame, ))[0]

    return imgInd, volFramePos


def debugAlignmentPlots(uvrDat,imgMetadat, imgInd, volFramePos):
    # figure to make some sanity check plots
    fig, axs = plt.subplots(1,2, figsize=(12,4))

    # sanity check if frame starts are dettected corrctly from analog signal
    axs[0].plot(np.arange(0,len(uvrDat.nidDf.imgfsig.values)), uvrDat.nidDf.imgfsig, '.-')
    axs[0].plot(np.arange(0,len(uvrDat.nidDf.imgfsig.values))[imgInd],
             uvrDat.nidDf.imgfsig[imgInd], 'r.')
    axs[0].set_xlim(1000,1200)
    axs[0].set_title('Sanity check 1:\nCheck if frame starts are detected correctly')
    utils.myAxisTheme(axs[0])

    # sanity check to see if time values align
    axs[1].plot(uvrDat.posDf.time.values[volFramePos],
             uvrDat.nidDf.timeinterp.values[imgInd[0::imgMetadat['fpv']]].astype('int') )
    axs[1].plot(uvrDat.posDf.time.values[volFramePos],uvrDat.posDf.time.values[volFramePos],'r')
    axs[1].axis('equal')
    axs[1].set_xlim(0,round(uvrDat.posDf.time.values[volFramePos][-1])+1)
    axs[1].set_ylim(0,round(uvrDat.nidDf.timeinterp.values[imgInd[0::imgMetadat['fpv']]].astype('int')[-1])+1)
    axs[1].set_title('Sanity check 2:\nCheck that time values align well')
    utils.myAxisTheme(axs[1])


# generate combined DataFrame
def combineImagingAndPosDf(imgDat, posDf, volFramePos):
    expDf = imgDat.copy()
    lendiff = len(expDf) - len(posDf.x.values[volFramePos])
    if lendiff != 0:
        print('Truncated fictrac recording.')
        expDf = expDf[:-lendiff]
    expDf['posTime'] = posDf.time.values[volFramePos]
    expDf['x'] = posDf.x.values[volFramePos]
    expDf['y'] = posDf.y.values[volFramePos]
    expDf['angle'] = posDf.angle.values[volFramePos]
    try:
        expDf['vT'] = posDf.vT.values[volFramePos]
        expDf['vR'] = posDf.vR.values[volFramePos]
        expDf['vTfilt'] = posDf.vT_filt.values[volFramePos]
        expDf['vRfilt'] = posDf.vR_filt.values[volFramePos]
    except AttributeError:
        from unityvr.analysis import posAnalysis
        posDf = posAnalysis.computeVelocities(posDf)
        expDf['vT'] = posDf.vT.values[volFramePos]
        expDf['vR'] = posDf.vR.values[volFramePos]
        expDf['vTfilt'] = posDf.vT_filt.values[volFramePos]
        expDf['vRfilt'] = posDf.vR_filt.values[volFramePos]
    return expDf


def loadAndAlignPreprocessedData(root, subdir, flies, conditions, trials, panDefs, condtype, img = 'img', vr = 'uvr'):
    allExpDf = pd.DataFrame()
    for f, fly in enumerate(flies):

        print(fly)
        flyStatsdf = pd.DataFrame(columns=['fly','condition','circmean','circvar','circvarPVA','circmeanCorr'])
        flyStats = np.ones((4, len(conditions)))*np.nan
        condlabel = []
        for c, cond in enumerate(conditions):

            for t, trial in enumerate(trials):
                preprocDir = sep.join([root,'preproc',subdir, fly, cond, trial])
                try:
                    imgDat = pd.read_csv(sep.join([preprocDir, img,'roiDFF.csv'])).drop(columns=['Unnamed: 0'])
                except FileNotFoundError:
                    print('missing file')
                    continue

                with open(sep.join([preprocDir, img,'imgMetadata.json'])) as json_file:
                    imgMetadat = json.load(json_file)

                uvrDat = logproc.loadUVRData(sep.join([preprocDir, vr]))
                posDf = uvrDat.posDf

                imgInd, volFramePos = findImgFrameTimes(uvrDat,imgMetadat)
                expDf = combineImagingAndPosDf(imgDat, posDf, volFramePos)

                if 'B2s' in panDefs.getPanID(cond) and condtype == '2d':
                    expDf['angleBrightAligned'] = np.mod(expDf['angle'].values-0*180/np.pi,360)
                else:
                    expDf['angleBrightAligned'] = np.mod(expDf['angle'].values-panDefs.panOrigin[panDefs.getPanID(cond)]*180/np.pi,360)
                    xr, yr = utils.rotatepath(expDf.x.values,expDf.y.values, -panDefs.panOrigin[panDefs.getPanID(cond)])
                    expDf.x = xr
                    expDf.y = yr
                #expDf['flightmask'] = np.logical_and(expDf.vTfilt.values < maxVt, expDf.vTfilt.values > minVt)
                expDf['fly'] = fly
                expDf['condition'] = cond
                expDf['trial'] = trial

                allExpDf = pd.concat([allExpDf,expDf])
    return allExpDf
