# Batch processing function to preprocess unity logs from a single fly experiment

def preprocessUnityLogs(rootDir, dataDir):
    import numpy as np
    import matplotlib.pyplot as plt
    from os import listdir
    from os.path import isfile, join

    from unityvr.preproc import logproc as lp

    #rootDir = '/Volumes/jayaramanlab/Hannah/Projects/FlyVR2P/Data/'
    #dataDir = 'disappearingSun/SS96_x_7f/EB/f04'

    dirName = rootDir + 'raw/' + dataDir
    condition = datPath.split('/')[3]
    preprocDir = rootDir + 'preproc/'+ dataDir

    fileNames = [f for f in listdir(dirName) if isfile(join(dirName, f)) and '.json' in f]
    print(fileNames)
    
    for fileName in fileNames:
    uvrTrial = lp.constructUnityVRexperiment(dirName,fileName)
    uvrTrial.printMetadata()

    savepath = uvrTrial.saveData(preprocDir, (uvrTrial.metadata['expid']).split('_')[-1] + '/' + uvrTrial.metadata['trial'])
    print(savepath)
