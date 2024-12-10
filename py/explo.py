import numpy as np
import pandas as pd

import sys
import os

cdir = os.getcwd() #get current directory
os.chdir(cdir) #make sure you are in the current directory
sys.path.append(cdir) #append directory to enable import of functions from current directory files

from integration import run
from funcs import getAvgPSD, getAvgSpaceFrequ


# # # # # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # # # # # # # # #
# # # - - - - - - - - - - - - - - - - - - - - naive Ansatz to identify pattern type - - - - - - - - - - - - - - - - - - # # #
# # # # # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # # # # # # # # #


def collectPatterns(fp, params, maxfreq=300, nperseg=1, ue=None):
    
    """ This function collects the type of activity-pattern that is shown after running a simulation for different settings of parameters 
    (fix given by params, varied in trng-df DataFrame) initialized in each available fixed point per parametrization. 
    Pattern-Identification on basis of frequency over space and over time.
    
    INPUT:
    :fp: fixed point for initialisation 
    :params: dictionary of fix parameters
    :max_freq: integer of maximal frequency to return
    :nperseg: window-size for average PSD computation
    :ue: already (transient time) cut-off activity array of params.n many nodes
    
    OUTPUT:
    :pattern: type of the emerging pattern after initialising the model in the corresponding fixed point.
        stationary=1
        temporal=2
        spatial=3
        spatiotemporal=4
        e.g. parametrization shows 3 fixed points, [fp1, fp2, fp3], init in fp1 shows spatial, in fp2 &fp3 stationary patterns => patterns=[3,1,1]
    :temporal_frequency, spatial_frequency: dominant temporal/spatial frequency of ue (where the corrsponding PSD has maximum power), floats
    :temporal_frequency_std, spatial_frequency_std: standard deviation of dominant temporal/spatial frequency of ue, floats
    """
        
    if np.any(ue)!=None:
        ue = ue
    else:
        ue, _ = run(params, itype='rungekutta', fp=fp)

    fs = (1000 * (1/params.dt)) #temporal sampling frequency
    
    #collect the type of pattern based on the last 3 seconds of simulation time
    ## this saves computation time which takes a while if one has to do 
    ## a ton of data for, e.g., an entire minute even though there ain't any oscillations
    #all space frequencies occuring, dominant spatial frequency, average power spectrum for pattern-detection
    fx_all, fx, fx_psd = getAvgSpaceFrequ(ue.T[:,-int(3 * 1000 * (1 / params['dt'])):], fs=params['n'], nperseg=1)
    #all temporal frequencies occuring, temporal average power spectrum
    ft_all, ft_psd = getAvgPSD(ue[:,-int(3 * 1000 * (1 / params['dt'])):], fs=fs, maxfreq=maxfreq, nperseg=3)
    
    #collect dominant temporal frequency
    ft = ft_all[np.argmax(ft_psd)]
    print(ft, fs)
    
    #check if temporally stationary
    temporal_threshold = (2*params.dt) / nperseg
    temporally_stationary = any((ft < temporal_threshold, all(ft_psd <= 1e-4)))
    
    #check if spatially stationary
    spatially_stationary = any((fx <= 1, all(fx_psd <= 1e-4)))
    
    
    if spatially_stationary and temporally_stationary:
        pattern = 1
        #no oscillations, no feature acquisition
        return pattern, 0, 0, 0, 0
    elif spatially_stationary and not temporally_stationary:
        pattern = 2
        #collect ft for entire data, not only last 3s - neglect spatial features since it has none
        ft_all, ft_psd = getAvgPSD(ue, fs=fs, maxfreq=maxfreq, nperseg=nperseg)
        ft = ft_all[np.argmax(ft_psd)]
        print(ft)
        ft_std = np.std(ft_psd, ddof=1)
        return pattern, ft, ft_std, 0, 0
    elif not spatially_stationary and temporally_stationary:
        pattern = 3
        #collect fx for entire data not only last 3s - neglect temporal features since there ain't none
        fx_all, fx, fx_psd = getAvgSpaceFrequ(ue.T, fs=params['n'], nperseg=1)
        fx_std = np.std(fx_all, ddof=1)
        return pattern, 0, 0, fx, fx_std
    else:
        #spatiotemporal pattern
        pattern = 4
        
        #collect ft for entire data, not only last 3s
        ft_all, ft_psd = getAvgPSD(ue, fs=fs, maxfreq=maxfreq, nperseg=nperseg)
        ft = ft_all[np.argmax(ft_psd)]
        ft_std = np.std(ft_psd, ddof=1)
        
        #collect fx for entire data not only last 3s
        fx_all, fx, fx_psd = getAvgSpaceFrequ(ue.T, fs=params['n'], nperseg=1)
        fx_std = np.std(fx_all, ddof=1)
        
        return pattern, ft, ft_std, fx, fx_std
