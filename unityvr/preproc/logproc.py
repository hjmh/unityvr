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
nidDfCols = ['frame','time','dt','pdsig']

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
    
    
    def saveData(self, saveDir):
        
        saveName = '_'.join([self.metadata['genotype'],'fly'+str(self.metadata['flyid']),
                             self.metadata['date'],self.metadata['time']])
        
        saveDir = sep.join([saveDir,self.metadata['expid']])
        # make directory
        if not exists(sep.join([saveDir,saveName])):
            makedirs(sep.join([saveDir,saveName]))
            
            
        # save metadata
        with open(sep.join([saveDir,saveName,'metadata.json']), 'w') as outfile:
            json.dumps(self.metadata, indent=4)
        
        # save dataframes
        self.objDf.to_csv(sep.join([saveDir,saveName,'objDf.csv']))
        self.posDf.to_csv(sep.join([saveDir,saveName,'posDf.csv']))
        self.ftDf.to_csv(sep.join([saveDir,saveName,'ftDf.csv']))
        self.nidDf.to_csv(sep.join([saveDir,saveName,'nidDf.csv']))

        
# constructor for unityVRexperiment
def constructUnityVRexperiment(dirName,fileName):
    
    dat = openUnityLog(dirName, fileName)
    
    metadat = makeMetaDict(dat, fileName)
    objDf = objDfFromLog(dat)
    posDf, ftDf, nidDf = timeseriesDfFromLog(dat)

    uvrexperiment = unityVRexperiment(metadata=metadat,posDf=posDf,ftDf=ftDf,nidDf=nidDf,objDf=objDf)
    
    return uvrexperiment


def parseHeader(notes, headerwords, metadat):
    
    for i, hw in enumerate(headerwords[:-1]):
        if hw in notes:
            metadat[i] = notes[notes.find(hw)+len(hw)+1:notes.find(headerwords[i+1])].split('-')[0]
            
    return metadat

def makeMetaDict(dat, fileName):
    headerwords = ["expid", "experiment", "genotype","flyid","sex","notes","\n"]
    metadat = ['testExp', 'test experiment', 'testGenotype', 'NA', 'NA', "NA"]
    
    if 'headerNotes' in dat[0].keys():
        headerNotes = dat[0]['headerNotes']
        metadat = parseHeader(headerNotes, headerwords, metadat)

    [datestr, timestr] = fileName.split('.')[0].split('_')[1:]
    
    metadata = {
        'expid': metadat[0].strip(),
        'experiment': metadat[1].strip(),
        'genotype': metadat[2].strip(),
        'sex': metadat[4].strip(),
        'flyid': metadat[3].strip(),
        'date': datestr,
        'time': timestr,
        'notes': metadat[5].strip()
    }
    
    return metadata
    
    
def openUnityLog(dirName, fileName):
    '''load json log file'''
    import json
    from os.path import sep
    
    # Opening JSON file 
    f = open(sep.join([dirName, fileName]),) 

    # returns JSON object as  
    # a dictionary 
    data = json.load(f)
    
    return data


# Functions for extracting data from log file and converting it to pandas dataframe

def objDfFromLog(dat):
    objDf = pd.DataFrame(columns=objDfCols)

    nlines = sum(1 for line in dat)

    for l in range(nlines):
        if 'data' in dat[l].keys(): 
            line = dat[l]['data']
        else:
            line = dat[l]
        if('meshGameObjectPath' in line.keys()):
            framedat = {'name': line['meshGameObjectPath'], 
                        'collider': line['colliderType'], 
                        'px': line['worldPosition']['x'], 
                        'py': line['worldPosition']['z'],
                        'pz': line['worldPosition']['y'],
                        'rx': line['worldRotationDegs']['x'], 
                        'ry': line['worldRotationDegs']['z'],
                        'rz': line['worldRotationDegs']['y'],
                        'sx': line['worldScale']['x'], 
                        'sy': line['worldScale']['z'],
                        'sz': line['worldScale']['y']}
            objDf = objDf.append(framedat, ignore_index = True)
            
    return objDf

def timeseriesDfFromLog(dat):
    from scipy.signal import medfilt

    posDf = pd.DataFrame(columns=posDfCols)
    ftDf = pd.DataFrame(columns=ftDfCols)
    dtDf = pd.DataFrame(columns=dtDfCols)
    pdDf = pd.DataFrame(columns = ['frame','time','pdsig'])
    # add 3rd timeseries with deltaTime? Or merge onto one?

    nlines = sum(1 for line in dat)
    
    for l in range(nlines):
        line = dat[l]
            
        if( 'worldPosition' in line.keys() and not 'meshGameObjectPath' in line.keys() ):
            framedat = {'frame': line['frame'], 
                        'time': line['timeSecs'], 
                        'x': line['worldPosition']['x'], 
                        'y': line['worldPosition']['z'],
                        'angle': line['worldRotationDegs']['y'],
                        'dx':line['actualTranslation']['x'],
                        'dy':line['actualTranslation']['z'],
                        'dxattempt': line['attemptedTranslation']['x'],
                        'dyattempt': line['attemptedTranslation']['z']
                       }
            posDf = posDf.append(framedat, ignore_index = True)
            
        if( 'ficTracDeltaRotationVectorLab' in line.keys() ):
            framedat = {'frame': line['frame'], 
                        'ficTracTReadMs': line['ficTracTimestampReadMs'], 
                        'ficTracTWriteMs': line['ficTracTimestampWriteMs'], 
                        'dx': line['ficTracDeltaRotationVectorLab']['x'], 
                        'dy': line['ficTracDeltaRotationVectorLab']['y'],
                        'dz': line['ficTracDeltaRotationVectorLab']['z']}
            ftDf = ftDf.append(framedat, ignore_index = True)
            
        if( 'deltaTime' in line.keys() ):
            framedat = {'frame': line['frame'], 
                        'time': line['timeSecs'], 
                        'dt': line['deltaTime']}
            dtDf = dtDf.append(framedat, ignore_index = True)
        
        if( 'tracePD' in line.keys() ):
            framedat = {'frame': line['frame'], 
                        'time': line['timeSecs'], 
                        'pdsig': line['tracePD']}
            pdDf = pdDf.append(framedat, ignore_index = True)
            
    posDf.time = posDf.time-posDf.time[0]
    dtDf.time = dtDf.time-dtDf.time[0]
    pdDf.time = pdDf.time-pdDf.time[0]
    
    ftDf.ficTracTReadMs = ftDf.ficTracTReadMs-ftDf.ficTracTReadMs[0]
    ftDf.ficTracTWriteMs = ftDf.ficTracTWriteMs-ftDf.ficTracTWriteMs[0]
    
    posDf = pd.merge(dtDf, posDf, on="frame", how='outer').rename(columns={'time_x':'time'}).drop(['time_y'],axis=1)
    nidDf = pd.merge(dtDf, pdDf, on="frame", how='outer').rename(columns={'time_x':'time'}).drop(['time_y'],axis=1)

    
    nidDf["pdFilt"]  = nidDf.pdsig.values
    nidDf.pdFilt[np.isfinite(nidDf.pdsig)] = medfilt(nidDf.pdsig[np.isfinite(nidDf.pdsig)])
    nidDf["pdThresh"]  = 1*(np.asarray(nidDf.pdFilt>=np.nanmedian(nidDf.pdFilt.values)))
    
    nidDf = generateInterTime(nidDf)
    
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