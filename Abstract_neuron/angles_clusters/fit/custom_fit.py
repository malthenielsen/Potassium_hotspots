import numpy as np
from matplotlib import pyplot as plt
plt.style.use('science')

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('science')
from glob import glob
import os
from scipy import stats
from scipy.signal import find_peaks
from iminuit import Minuit
import sys                                             # Module to see files and folders in directories

from ExternalFunctions import UnbinnedLH, BinnedLH, Chi2Regression
from ExternalFunctions import nice_string_output, add_text_to_ax   # Useful functions to print fit results on figure

def R2(y, fy):
    ybar = np.mean(y)
    SS_RES = np.sum((y - f)**2)
    SS_TOT = np.sum((y - ybar)**2)
    return 1 - SS_RES/SS_TOT

def func(path):

    fnames_soma = glob(f'{path}/soma?*.npy')
    fnames_trunk = glob(f'{path}/angle?*.npy')
    fnames_trunk.sort(key=os.path.getctime)
    fnames_soma.sort(key=os.path.getctime)

    data = np.zeros((3,20,len(fnames_trunk)))
    angle = np.zeros((20,len(fnames_trunk)))


    for l, fname in enumerate(fnames_soma):
        print(fname)
        soma = np.load(fname)
        print(soma.shape)
        FPS_arr = np.zeros(20*3)
        v_arr = np.zeros(20*3)
        bin_ = []
        for i in range(len(soma)):
            inds, _ = find_peaks(soma[i], height = -0)
            FPS_arr[i]= len(inds)
        data[:,:,l] = FPS_arr.reshape(20,3).T

    for l, fname in enumerate(fnames_trunk):
        angle[:,l] = np.load(fname).reshape(20)


    #  mean = np.mean(data, axis = 2)
    #  eom = np.std(data, axis = 2)/np.sqrt(data.shape[2])
    return data.reshape(3,-1), angle.ravel()


mean, angles = func('./gain')

fig, ax = plt.subplots(1,1, figsize = (8,6))
#  for i in range(2,3):
fps = np.linspace(0,15,100)
ax.scatter(angles, mean[1,:], alpha = .05)
#  ax.plot(fps, fps*1.05)
#  ax.plot(fps, fps+.5)

plt.show()

def mfunc(x,a):
    return a*x

def afunc(x,a):
    return x + a


for i in range(1,3):
    chi2_fit = Chi2Regression(mfunc, mean[0,:], mean[i,:])
    minuit_chi2 = Minuit(chi2_fit, a = 1.1)
    minuit_chi2.errordef = 1
    minuit_chi2.migrad()
    print(minuit_chi2.values[:], 'a')
    print(minuit_chi2.errors[:])
    print(minuit_chi2.fval)
    print(stats.chi2.sf(minuit_chi2.fval, len(mean[0,:]) - 1), 'p mul')
    f = mfunc(mean[0,:], minuit_chi2.values['a'])
    print(R2(mean[i,:],f)) 


    chi2_fit = Chi2Regression(afunc, mean[0,:], mean[i,:])
    minuit_chi2 = Minuit(chi2_fit, a = 1)
    minuit_chi2.errordef = 1
    minuit_chi2.migrad()
    print(minuit_chi2.values[:])
    print(minuit_chi2.errors[:])
    print(minuit_chi2.fval)
    print(stats.chi2.sf(minuit_chi2.fval, len(mean[0,:]) - 1), 'p add')
    f = afunc(mean[0,:], minuit_chi2.values['a'])
    print(R2(mean[i,:],f)) 


plt.show()


