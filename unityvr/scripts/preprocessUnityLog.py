#!/usr/bin/python
# Batch processing function to preprocess unity logs from a single fly experiment
from os import listdir
from os.path import isfile, join
from unityvr.preproc import logproc as lp
import sys


def preprocessUnityVRlogs(rootDir, dataDir):
    dirName = rootDir + 'raw/' + dataDir
    preprocDir = rootDir + 'preproc/'+ dataDir
    fileNames = [f for f in listdir(dirName) if isfile(join(dirName, f)) and '.json' in f]

    for fileName in fileNames:
        print(fileName  + '\n')
        savepath =  lp.convertJsonToPandas(dirName,fileName,preprocDir, computeNiDf=False)
        #uvrTrial = lp.constructUnityVRexperiment(dirName,fileName)
        #uvrTrial.printMetadata()
        #savepath = uvrTrial.saveData(preprocDir, (uvrTrial.metadata['expid']).split('_')[-1] + '/' + uvrTrial.metadata['trial'])
        print(savepath)

if __name__ == "__main__":
    # get command line argument
    if len(sys.argv) < 3:
        print('Please specify a root directory where to find a folder with raw data "raw" and one for the preprocessed data "preproc".\
        As second argument, provide a relative path to the data directory within root/raw')
        #Example arguments
        #rootDir = '/Volumes/jayaramanlab/Hannah/Projects/FlyVR2P/Data/'
        #dataDir = 'disappearingSun/SS96_x_7f/EB/f04'
    else:
        print(sys.argv[1] + '\n')
        print(sys.argv[2]+ '\n')
        preprocessUnityVRlogs(sys.argv[1],sys.argv[2])

    print("\n all done!")
