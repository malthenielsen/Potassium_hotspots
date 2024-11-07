import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
from glob import glob
import os
from scipy import stats
from scipy.signal import find_peaks
#  from ExternalFunctions import Chi2Regression
from appstatpy.ExternalFunctions import *
from iminuit import Minuit
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 14})
plt.rcParams['svg.fonttype'] = 'none'

def rm(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')

trunk_lengths = np.arange(50, 1000, 25)
N_t = len(trunk_lengths)
print(N_t)

def func(path):
    fnames_trunk = glob(f'{path}/soma?*.npy')
    fnames_trunk.sort(key=os.path.getctime)
    print(fnames_trunk)


    colors = ['darkgrey', 'dodgerblue', 'goldenrod']

    data = np.zeros((3,N_t,len(fnames_trunk)))


    for l, fname in enumerate(fnames_trunk):
        trunk = np.load(fname)
        bin_ = []
        
        for i in range(len(trunk)):
            inds, _ = find_peaks(trunk[i], height = -30)

            if i%3 == 0:
                baseline = np.mean(trunk[i][3500:4000])
                fig, ax = plt.subplots(1,1, figsize = (7,6))
            trunk_tmp = trunk[i][5000:30000] - baseline
            #  plt.plot(trunk[i][:] - baseline)
            if i%3 == 2:
                plt.show()

            area = np.trapezoid(trunk_tmp, dx = 1/40)
            bin_.append(area)
        bin_ = np.array(bin_)
        data[:,:,l] = bin_.reshape(N_t,3).T


    mean = np.mean(data, axis = 2)
    eom = np.std(data,axis =2)/np.sqrt(data.shape[-1])
    return mean, eom 

paths = ['./trunk_test_scaled/current_large/folder_trunk_test/']#,
mean, eom = func(paths[0])







