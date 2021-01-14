### This module contains basic preprocessing functions for processing the unity VR log file

import pandas as pd

#dataframe column defs
objDfCols = ['name','collider','px','py','pz','rx','ry','rz','sx','sy','sz']

posDfCols = ['frame','time','dt','x','y','angle']
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

# TODO: merge all functions below to construct dfs in parallel

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

def posDfFromLog(dat):    
    posDf = pd.DataFrame(columns=posDfCols)
    
    nlines = sum(1 for line in dat)
    
    for l in range(nlines):
        if 'data' in dat[l].keys(): 
            line = dat[l]['data']
        else:
            line = dat[l]
        if( 'worldPosition' in line.keys() and not 'meshGameObjectPath' in line.keys() ):
            framedat = {'frame': dat[l]['frame'], 
                        'time': dat[l]['timeSecs'], 
                        'dt': dat[l]['deltaTime'], 
                        'x': line['worldPosition']['x'], 
                        'y': line['worldPosition']['z'],
                        'angle': line['worldRotationDegs']['y'],
                        'dx':line['actualTranslation']['x'],
                        'dy':line['actualTranslation']['z'],
                        'dxattempt': line['attemptedTranslation']['x'],
                        'dyattempt': line['attemptedTranslation']['z']
                       }
            posDf = posDf.append(framedat, ignore_index = True)
    posDf.time = posDf.time-posDf.time[0]
    return posDf


def ftDfFromLog(dat):
    ftDf = pd.DataFrame(columns=ftDfCols)
    
    nlines = sum(1 for line in dat)
    for l in range(nlines):
        if 'data' in dat[l].keys(): 
            line = dat[l]['data']
        else:
            line = dat[l]
        if( 'ficTracDeltaRotationVectorLab' in line.keys() ):
            framedat = {'frame': dat[l]['frame'], 
                        'ficTracTReadMs': line['ficTracTimestampReadMs'], 
                        'ficTracTWriteMs': line['ficTracTimestampWriteMs'], 
                        'dx': line['ficTracDeltaRotationVectorLab']['x'], 
                        'dy': line['ficTracDeltaRotationVectorLab']['y'],
                        'dz': line['ficTracDeltaRotationVectorLab']['z']}
            ftDf = ftDf.append(framedat, ignore_index = True)
    ftDf.ficTracTReadMs = ftDf.ficTracTReadMs-ftDf.ficTracTReadMs[0]
    ftDf.ficTracTWriteMs = ftDf.ficTracTWriteMs-ftDf.ficTracTWriteMs[0]
    return ftDf