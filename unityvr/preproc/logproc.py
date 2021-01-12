### This module contains basic preprocessing functions for processing the unity VR log file

import pandas as pd

def openUnityLog(dirName, fileName):
    
    import json
    from os.path import sep
    
    # Opening JSON file 
    f = open(sep.join([dirName, fileName]),) 

    # returns JSON object as  
    # a dictionary 
    data = json.load(f)
    
    return data