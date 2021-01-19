### This module contains basic preprocessing functions for processing the unity VR log file

import pandas as pd
from dataclasses import dataclass, asdict
from os import mkdir, makedirs
from os.path import sep, isfile, exists
import json

#dataframe column defs
objDfCols = ['name','collider','px','py','pz','rx','ry','rz','sx','sy','sz']

posDfCols = ['frame','time','x','y','angle']
ftDfCols = ['frame','ficTracTReadMs','ficTracTWriteMs','dx','dy','dz']

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
    posDf = pd.DataFrame(columns=posDfCols)
    ftDf = pd.DataFrame(columns=ftDfCols)
    
    # add 3rd timeseries with deltaTime? Or merge onto one?

    nlines = sum(1 for line in dat)
    
    for l in range(nlines):
        if 'data' in dat[l].keys(): 
            line = dat[l]['data']
        else:
            line = dat[l]
            
        if( 'worldPosition' in line.keys() and not 'meshGameObjectPath' in line.keys() ):
            framedat = {'frame': dat[l]['frame'], 
                        'time': dat[l]['timeSecs'], 
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
            framedat = {'frame': dat[l]['frame'], 
                        'ficTracTReadMs': line['ficTracTimestampReadMs'], 
                        'ficTracTWriteMs': line['ficTracTimestampWriteMs'], 
                        'dx': line['ficTracDeltaRotationVectorLab']['x'], 
                        'dy': line['ficTracDeltaRotationVectorLab']['y'],
                        'dz': line['ficTracDeltaRotationVectorLab']['z']}
            ftDf = ftDf.append(framedat, ignore_index = True)
            
    posDf.time = posDf.time-posDf.time[0]
    
    ftDf.ficTracTReadMs = ftDf.ficTracTReadMs-ftDf.ficTracTReadMs[0]
    ftDf.ficTracTWriteMs = ftDf.ficTracTWriteMs-ftDf.ficTracTWriteMs[0]
    
    return posDf, ftDf


# Data class definition

objDfCols = ['name','collider','px','py','pz','rx','ry','rz','sx','sy','sz']

posDfCols = ['frame','time','x','y','angle']
ftDfCols = ['frame','ficTracTReadMs','ficTracTWriteMs','dx','dy','dz']

@dataclass
class unityVRexperiment:

    # metadata as dict
    metadata: dict
        
    imaging: bool = False
    brainregion: str = None
    
    # timeseries data
    posDf: pd.DataFrame = pd.DataFrame(columns=posDfCols)
    ftDf: pd.DataFrame = pd.DataFrame(columns=ftDfCols)
        
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

        
# constructor for unityVRexperiment
def constructUnityVRexperiment(dirName,fileName):
    
    dat = openUnityLog(dirName, fileName)
    
    metadat = makeMetaDict(dat, fileName)
    objDf = objDfFromLog(dat)
    posDf,ftDf = timeseriesDfFromLog(dat)

    uvrexperiment = unityVRexperiment(metadata=metadat,posDf=posDf,ftDf=ftDf,objDf=objDf)
    
    return uvrexperiment


def parseHeader(notes):
    headerwords = ["expid", "experiment", "genotype","flyid","sex","\n"]
    metadat = ['testExp', 'test experiment', 'testGenotype', 'NA', 'NA']
    
    for i, hw in enumerate(headerwords[:-1]):
        if hw in notes:
            metadat[i] = notes[notes.find(hw)+len(hw)+1:notes.find(headerwords[i+1])].split('-')[0]
    expid=metadat[0].strip()
    exp=metadat[1].strip()
    gt=metadat[2].strip()
    fi=metadat[3].strip()
    sx=metadat[4].strip()
    
    return expid, exp, gt, fi, sx

def makeMetaDict(dat, fileName):
    
    if 'headerNotes' in dat[0].keys():
        headerNotes = dat[0]['headerNotes']
        [expid, exp, gt, fi ,sx] = parseHeader(headerNotes)
    else:
        expid = 'testExp'
        exp = 'test experiment'
        gt = 'testGenotype'
        fi = 'NA'
        sx = 'NA'

    [datestr, timestr] = fileName.split('.')[0].split('_')[1:]
    
    metadata = {
        'expid': expid,
        'experiment': exp,
        'genotype': gt,
        'sex': sx,
        'flyid': fi,
        'date': datestr,
        'time': timestr
    }
    
    return metadata
    