### This module contains basic preprocessing functions for processing the unity VR log file

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from os import mkdir, makedirs
from os.path import sep, isfile, exists
import json

#dataframe column defs
objDfCols = ['name','collider','px','py','pz','rx','ry','rz','sx','sy','sz']

posDfCols = ['frame','time','x','y','angle']
ftDfCols = ['frame','ficTracTReadMs','ficTracTWriteMs','dx','dy','dz']
dtDfCols = ['frame','time','dt']
nidDfCols = ['frame','time','dt','pdsig','imgfsig']
texDfCols = ['frame','time','xtex','ytex']
# Data class definition

@dataclass
class unityVRexperiment:

    # metadata as dict
    metadata: dict

    imaging: bool = False
    brainregion: str = None

    # timeseries data
    posDf: pd.DataFrame = pd.DataFrame(columns=posDfCols)
    ftDf: pd.DataFrame = pd.DataFrame(columns=ftDfCols)
    nidDf: pd.DataFrame = pd.DataFrame(columns=nidDfCols)
    texDf: pd.DataFrame = pd.DataFrame(columns=texDfCols)
    shapeDf: pd.DataFrame = pd.DataFrame()
    timeDf: pd.DataFrame = pd.DataFrame()

    # object locations
    objDf: pd.DataFrame = pd.DataFrame(columns=objDfCols)

    # methods
    def printMetadata(self):
        print('Metadata:\n')
        for key in self.metadata:
            print(key, ' : ', self.metadata[key])

    ## data wrangling
    def downsampleftDf(self):
        frameftDf = self.ftDf.groupby("frame").sum()
        frameftDf.reset_index(level=0, inplace=True)
        return frameftDf

    def saveData(self, saveDir, saveName):
        savepath = sep.join([saveDir,saveName,'uvr'])

        # make directory
        if not exists(savepath):
            makedirs(savepath)

        # save metadata
        with open(sep.join([savepath,'metadata.json']), 'w') as outfile:
            json.dump(self.metadata, outfile,indent=4)

        # save dataframes
        self.objDf.to_csv(sep.join([savepath,'objDf.csv']))
        self.posDf.to_csv(sep.join([savepath,'posDf.csv']))
        self.ftDf.to_csv(sep.join([savepath,'ftDf.csv']))
        self.nidDf.to_csv(sep.join([savepath,'nidDf.csv']))
        self.texDf.to_csv(sep.join([savepath,'texDf.csv']))
        self.shapeDf.to_csv(sep.join([savepath,'shapeDf.csv']))
        self.timeDf.to_csv(sep.join([savepath,'timeDf.csv']))

        return savepath

# constructor for unityVRexperiment
def constructUnityVRexperiment(dirName,fileName,imaging=False,test=False):

    dat = openUnityLog(dirName, fileName)

    metadat = makeMetaDict(dat, fileName)
    objDf = objDfFromLog(dat)
    posDf, ftDf, nidDf = timeseriesDfFromLog(dat)
    texDf = texDfFromLog(dat)

    uvrexperiment = unityVRexperiment(metadata=metadat,posDf=posDf,ftDf=ftDf,nidDf=nidDf,objDf=objDf,texDf=texDf)

    return uvrexperiment


def loadUVRData(savepath):

    with open(sep.join([savepath,'metadata.json'])) as json_file:
        metadat = json.load(json_file)
    objDf = pd.read_csv(sep.join([savepath,'objDf.csv'])).drop(columns=['Unnamed: 0'])
    # ToDo remove when Shivam has removed fixation values from posDf
    posDf = pd.read_csv(sep.join([savepath,'posDf.csv']),dtype={'fixation': 'string'}).drop(columns=['Unnamed: 0','fixation'],errors='ignore')
    ftDf = pd.read_csv(sep.join([savepath,'ftDf.csv'])).drop(columns=['Unnamed: 0'])
    nidDf = pd.read_csv(sep.join([savepath,'nidDf.csv'])).drop(columns=['Unnamed: 0'])


    try:
        texDf = pd.read_csv(sep.join([savepath,'texDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        texDf = pd.DataFrame()
        #No texture mapping time series was recorded with this experiment, fill with empty DataFrame

    try:
        shapeDf = pd.read_csv(sep.join([savepath,'shapeDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        shapeDf = pd.DataFrame()
        #Shape dataframe was not computed. Fill with empty DataFrame

    try:
        timeDf = pd.read_csv(sep.join([savepath,'timeDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        timeDf = pd.DataFrame()
        #Shape dataframe was not computed. Fill with empty DataFrame

    uvrexperiment = unityVRexperiment(metadata=metadat,posDf=posDf,ftDf=ftDf,nidDf=nidDf,objDf=objDf,texDf=texDf,shapeDf=shapeDf,timeDf=timeDf)

    return uvrexperiment


def parseHeader(notes, headerwords, metadat):

    for i, hw in enumerate(headerwords[:-1]):
        if hw in notes:
            metadat[i] = notes[notes.find(hw)+len(hw)+1:notes.find(headerwords[i+1])].split('~')[0].strip()

    return metadat

def makeMetaDict(dat, fileName):
    headerwords = ["expid", "experiment", "genotype","flyid","sex","notes","\n"]
    metadat = ['testExp', 'test experiment', 'testGenotype', 'NA', 'NA', "NA"]

    if 'headerNotes' in dat[0].keys():
        headerNotes = dat[0]['headerNotes']
        metadat = parseHeader(headerNotes, headerwords, metadat)

    [datestr, timestr] = fileName.split('.')[0].split('_')[1:3]

    matching = [s for s in dat if "ficTracBallRadius" in s]
    ballRad = matching[0]["ficTracBallRadius"]

    matching = [s for s in dat if "refreshRateHz" in s]
    setFrameRate = matching[0]["refreshRateHz"]

    metadata = {
        'expid': metadat[0],
        'experiment': metadat[1],
        'genotype': metadat[2],
        'sex': metadat[4],
        'flyid': metadat[3],
        'trial': 'trial'+fileName.split('.')[0].split('_')[-1][1:],
        'date': datestr,
        'time': timestr,
        'ballRad': ballRad,
        'setFrameRate': setFrameRate,
        'notes': metadat[5],
        'angle_convention':"right-handed"
    }

    return metadata


def openUnityLog(dirName, fileName):
    '''load json log file'''
    import json
    from os.path import sep

    # Opening JSON file
    f = open(sep.join([dirName, fileName]))

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    return data


# Functions for extracting data from log file and converting it to pandas dataframe

def objDfFromLog(dat):
    # get dataframe with info about objects in vr
    matching = [s for s in dat if "meshGameObjectPath" in s]
    entries = [None]*len(matching)
    for entry, match in enumerate(matching):
        framedat = {'name': match['meshGameObjectPath'],
                    'collider': match['colliderType'],
                    'px': match['worldPosition']['x'],
                    'py': match['worldPosition']['z'],
                    'pz': match['worldPosition']['y'],
                    'rx': match['worldRotationDegs']['x'],
                    'ry': match['worldRotationDegs']['z'],
                    'rz': match['worldRotationDegs']['y'],
                    'sx': match['worldScale']['x'],
                    'sy': match['worldScale']['z'],
                    'sz': match['worldScale']['y']}
        entries[entry] = pd.Series(framedat).to_frame().T
    objDf = pd.concat(entries,ignore_index = True)

    return objDf


def posDfFromLog(dat):
    # get info about camera position in vr
    matching = [s for s in dat if "attemptedTranslation" in s]
    entries = [None]*len(matching)
    for entry, match in enumerate(matching):
        framedat = {'frame': match['frame'],
                        'time': match['timeSecs'],
                        'x': match['worldPosition']['x'],
                        'y': match['worldPosition']['z'], #axes are named differently in Unity
                        'angle': (-match['worldRotationDegs']['y'])%360, #flip due to left handed convention in Unity
                        'dx':match['actualTranslation']['x'],
                        'dy':match['actualTranslation']['z'],
                        'dxattempt': match['attemptedTranslation']['x'],
                        'dyattempt': match['attemptedTranslation']['z']
                       }
        entries[entry] = pd.Series(framedat).to_frame().T
    posDf = pd.concat(entries,ignore_index = True)
    print('correcting for Unity angle convention.')

    return posDf


def ftDfFromLog(dat):
    # get fictrac data
    matching = [s for s in dat if "ficTracDeltaRotationVectorLab" in s]
    entries = [None]*len(matching)
    for entry, match in enumerate(matching):
        framedat = {'frame': match['frame'],
                        'ficTracTReadMs': match['ficTracTimestampReadMs'],
                        'ficTracTWriteMs': match['ficTracTimestampWriteMs'],
                        'dx': match['ficTracDeltaRotationVectorLab']['x'],
                        'dy': match['ficTracDeltaRotationVectorLab']['y'],
                        'dz': match['ficTracDeltaRotationVectorLab']['z']}
        entries[entry] = pd.Series(framedat).to_frame().T

    if len(entries) > 0:
        ftDf = pd.concat(entries, ignore_index = True)
    else:
        ftDf = pd.DataFrame()

    return ftDf

def dtDfFromLog(dat):
    # get delta time info
    matching = [s for s in dat if "deltaTime" in s]
    entries = [None]*len(matching)
    for entry, match in enumerate(matching):
        framedat = {'frame': match['frame'],
                    'time': match['timeSecs'],
                    'dt': match['deltaTime']}
        entries[entry] = pd.Series(framedat).to_frame().T
    dtDf = pd.concat(entries,ignore_index = True)

    return dtDf


def pdDfFromLog(dat):
    # get NiDaq signal
    matching = [s for s in dat if "tracePD" in s]
    entries = [None]*len(matching)
    for entry, match in enumerate(matching):
        framedat = {'frame': match['frame'],
                    'time': match['timeSecs'],
                    'pdsig': match['tracePD'],
                    'imgfsig': match['imgFrameTrigger']}
        entries[entry] = pd.Series(framedat).to_frame().T

    pdDf = pd.concat(entries,ignore_index = True)

    return pdDf

def texDfFromLog(dat):
    # get texture remapping log
    matching = [s for s in dat if "xpos" in s]
    if len(matching) == 0: return pd.DataFrame()

    entries = [None]*len(matching)
    for entry, match in enumerate(matching):
        framedat = {'frame': match['frame'],
                    'time': match['timeSecs'],
                    'xtex': match['xpos'],
                    'ytex': match['ypos']}
        entries[entry] = pd.Series(framedat).to_frame().T

    texDf = pd.concat(entries,ignore_index = True)

    texDf.time = texDf.time-texDf.time[0]

    return texDf

def ftTrajDfFromLog(directory, filename):
    cols = [14,15,16,17,18]
    colnames = ['x','y','heading','travelling','speed']
    ftTrajDf = pd.read_csv(directory+"/"+filename,usecols=cols,names=colnames)
    return ftTrajDf

def timeseriesDfFromLog(dat):
    from scipy.signal import medfilt

    posDf = pd.DataFrame(columns=posDfCols)
    ftDf = pd.DataFrame(columns=ftDfCols)
    dtDf = pd.DataFrame(columns=dtDfCols)
    pdDf = pd.DataFrame(columns = ['frame','time','pdsig', 'imgfsig'])

    posDf = posDfFromLog(dat)
    ftDf = ftDfFromLog(dat)
    dtDf = dtDfFromLog(dat)
    try:
        pdDf = pdDfFromLog(dat)
    except:
        print("No analog input data was recorded.")

    posDf.time = posDf.time-posDf.time[0]
    dtDf.time = dtDf.time-dtDf.time[0]
    if len(pdDf) > 0:
        pdDf.time = pdDf.time-pdDf.time[0]

    if len(ftDf) > 0:
        ftDf.ficTracTReadMs = ftDf.ficTracTReadMs-ftDf.ficTracTReadMs[0]
        ftDf.ficTracTWriteMs = ftDf.ficTracTWriteMs-ftDf.ficTracTWriteMs[0]
    else:
        print("No fictrac signal was recorded.")

    posDf = pd.merge(dtDf, posDf, on="frame", how='outer').rename(columns={'time_x':'time'}).drop(['time_y'],axis=1)

    if len(pdDf) > 0:
        nidDf = pd.merge(dtDf, pdDf, on="frame", how='outer').rename(columns={'time_x':'time'}).drop(['time_y'],axis=1)

        nidDf["pdFilt"]  = nidDf.pdsig.values
        nidDf.pdFilt.values[np.isfinite(nidDf.pdsig.values)] = medfilt(nidDf.pdsig.values[np.isfinite(nidDf.pdsig.values)])
        nidDf["pdThresh"]  = 1*(np.asarray(nidDf.pdFilt>=np.nanmedian(nidDf.pdFilt.values)))

        nidDf["imgfFilt"]  = nidDf.imgfsig.values
        nidDf.imgfFilt.values[np.isfinite(nidDf.imgfsig.values)] = medfilt(nidDf.imgfsig.values[np.isfinite(nidDf.imgfsig.values)])
        nidDf["imgfThresh"]  = 1*(np.asarray(nidDf.imgfFilt.values>=np.nanmedian(nidDf.imgfFilt.values)))

        nidDf = generateInterTime(nidDf)
    else:
        nidDf = posDf

    return posDf, ftDf, nidDf


def generateInterTime(tsDf):
    from scipy import interpolate

    tsDf['framestart'] = np.hstack([0,1*np.diff(tsDf.time)>0])

    tsDf['counts'] = 1
    sampperframe = tsDf.groupby('frame').sum()[['time','dt','counts']].reset_index(level=0)
    sampperframe['fs'] = sampperframe.counts/sampperframe.dt

    frameStartIndx = np.hstack((0,np.where(tsDf.framestart)[0]))
    frameStartIndx = np.hstack((frameStartIndx, frameStartIndx[-1]+sampperframe.counts.values[-1]-1))
    frameIndx = tsDf.index.values

    frameNums = tsDf.frame[frameStartIndx].values.astype('int')
    frameNumsInterp = np.hstack((frameNums, frameNums[-1]+1))

    timeAtFramestart = tsDf.time[frameStartIndx].values

    #generate interpolated frames
    frameinterp_f = interpolate.interp1d(frameStartIndx,frameNums)
    tsDf['frameinterp'] = frameinterp_f(frameIndx)

    timeinterp_f = interpolate.interp1d(frameStartIndx,timeAtFramestart)
    tsDf['timeinterp'] = timeinterp_f(frameIndx)

    return tsDf
