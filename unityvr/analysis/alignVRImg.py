# Functions for aligning imaging and VR data
import numpy as np
import matplotlib.pyplot as plt
from unityvr.viz import viz

def findImgFrameTimes(uvrDat,imgMetadat):

    imgInd = np.where(np.diff(uvrDat.nidDf.imgfsig.values)>3)[0]

    imgFrame = uvrDat.nidDf.frame.values[imgInd].astype('int')
    #print(imgFrame)

    #take only every x frame as start of volume
    volFrame = imgFrame[0::imgMetadat['fpv']]
    #print(len(volFrame))
    #print(len(imgDat.slice1))

    volFramePos = np.where(np.in1d(uvrDat.posDf.frame.values,volFrame, ))[0]
    #print(len(volFramePos))

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
    viz.myAxisTheme(axs[0])

    # sanity check to see if time values align
    axs[1].plot(uvrDat.posDf.time.values[volFramePos],
             uvrDat.nidDf.timeinterp.values[imgInd[0::imgMetadat['fpv']]].astype('int') )
    axs[1].plot(uvrDat.posDf.time.values[volFramePos],uvrDat.posDf.time.values[volFramePos],'r')
    axs[1].axis('equal')
    axs[1].set_xlim(0,round(uvrDat.posDf.time.values[volFramePos][-1])+1)
    axs[1].set_ylim(0,round(uvrDat.nidDf.timeinterp.values[imgInd[0::imgMetadat['fpv']]].astype('int')[-1])+1)
    axs[1].set_title('Sanity check 2:\nCheck that time values align well')
    viz.myAxisTheme(axs[1])

    return fig

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
