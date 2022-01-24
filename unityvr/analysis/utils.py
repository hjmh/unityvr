import pandas as pd
from os.path import sep, join

##generic utility functions

#function to carry attributes from one dataframe to another
def carryAttrs(df, posDf):
    attrs = list(posDf.__dict__)[5:]
    for a in attrs:
        df.__dict__[a] = posDf.__dict__[a]

    return df

#function to append metadata to figure name
def getTrajFigName(figurename,saveDir,metadata):
    #appends metadata to figure name
    name = saveDir+sep+'_'.join([figurename, metadata['genotype'],
                                      metadata['sex'],
                                      metadata['flyid'],
                                      metadata['expid'].split('_')[-1],
                                      metadata['trial']+'.pdf'])
    return name
