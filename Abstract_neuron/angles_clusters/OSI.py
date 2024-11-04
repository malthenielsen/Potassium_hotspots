import numpy as np
from matplotlib import pyplot as plt
#  plt.style.use('K_PAPER')
from glob import glob
import os
from scipy import stats, interpolate
from scipy.signal import find_peaks
from iminuit import Minuit
import sys                                             # Module to see files and folders in directories

from ExternalFunctions import UnbinnedLH, BinnedLH, Chi2Regression
from ExternalFunctions import nice_string_output, add_text_to_ax   # Useful functions to print fit results on figure

paths = ['./3frac_2575']
paths = ['./3frac_7525']

def func(path):
    fnames_soma = glob(f'{path}/soma?*.npy')
    fnames_trunk = glob(f'{path}/stack?*.npy')
    fnames_trunk.sort(key=os.path.getctime)
    fnames_soma.sort(key=os.path.getctime)
    #  x_val = np.linspace(0,90,200)
    #  angles = np.array([0,5,10,15,20,30,40,50,80,89])



    data = np.zeros((3,22,len(fnames_soma)))


    for l, fname in enumerate(fnames_soma):
        soma = np.load(fname)
        FPS_arr = np.zeros(22*3)
        for i in range(len(soma)):
            inds, _ = find_peaks(soma[i], height = 40)
            FPS_arr[i]= len(inds)
        data[:,:,l] = FPS_arr.reshape(22,3).T

    mean = np.mean(data, axis = 2)
    #  print(data)
    #  for i in range(3):
    #      cs = interpolate.CubicSpline(angles, mean[i,:])
    #      yp = cs(x_val)
    #      idx = np.argmin(np.abs(yp - .5*np.max(yp)))
    #      fwhm.append(x_val[idx])


    return mean, data

#  paths = ['./3frac_5050', './3frac_7525', './3frac_2575', './3frac_0100', './3frac_1000']
#  paths = ['./3frac_5050', './3frac_7525', './3frac_2575']
paths = ['./folder']
#  paths = ['./3frax_5050_test']

def alt_func(x, a):
    return a

def OSI(arr, ori):
    if np.mean(arr) == 0:
        return np.nan
    ori = np.deg2rad(ori)
    top = np.sum(arr * np.exp(2*1j*ori))
    F = top/np.sum(arr)
    return 1  - np.arctan2(np.imag(F), np.real(F))

#  angles = np.array([0,5,10,15,20,30,40,50,80,89])
#  angles = np.arange(0,90,5)
angles = np.linspace(0,35,18)
angles = np.append(angles, np.array([40,50,75,90]))
#  angles = np.arange(0,36,2)

fig, ax = plt.subplots(figsize = (5,6))
ax2 = ax.twinx()
#  color = ['teal', 'goldenrod', 'teal', '#75151E','#0047AB' ]
ls = ['solid', '--', '-.', ':', (0,(1,10))]
labels = ['50/50', '75/25', '25/75', '0/100', '100/0']
for idx , path in enumerate(paths):
    means, data = func(path)
    OSI_arr = np.zeros(3)
    for i in range(3):
        OSI_arr[i] = OSI(means[i,:], angles)
    x_arr = np.arange(3)
    ax2.scatter(x_arr, OSI_arr, ls = ls[idx], color = 'slategrey')
    ax2.plot(x_arr, OSI_arr, ls = ls[idx], color = 'slategrey', label = labels[idx])
    ax.scatter(x_arr, means[:,0], ls = ls[idx], color = 'tab:red')
    ax.plot(x_arr, means[:,0], ls = ls[idx], color = 'tab:red')
    #  ax2.scatter(x_arr, fwhm, ls = ls[idx], color = 'red')
    #  ax2.plot(x_arr,    fwhm, ls = ls[idx], color = 'red')
    #  print(means[:,0])
    #  print(idx)
ax2.set_ylim(0,1)
ax.set_ylim(12,25.5)
#  ax2.set_ylim(0,1)
#  ax.set_ylim(11,13.5)
ax.set_ylabel('Peak firing rate at $0^\circ$', color = 'tab:red')
ax2.set_ylabel('OSI', color = 'black')
ax2.set_xticks([0,1,2],['None', '$ Low \Delta E_K$', '$ High \Delta E_K$'] , rotation = 30)
ax2.legend(title = 'Ratio', loc = 4)
ax.text(-0.1, 1.05, 'D', fontweight = 'bold', transform = ax.transAxes, fontsize = 20)
#  ax.set_yticks(np.arange(11.5, 14, .5), np.arange(11.5, 13.5, .5))
#  ax2.set_yticks(np.arange(.6, .85, .05), np.round(np.arange(.6, .85, .05), 2))
fig.savefig('OSI', dpi =  200)
fig.savefig('FIG_4D.svg', dpi =  400)
fig.savefig('FIG_4D.pdf', dpi =  400)

fig, ax = plt.subplots(figsize = (5,6))
ax2 = ax.twinx()
#  color = ['teal', 'goldenrod', 'teal', '#75151E','#0047AB' ]
ls = ['solid', '--', '-.', ':', (0,(1,10))]
labels = ['50/50', '75/25', '25/75', '0/100', '100/0']
for idx , path in enumerate(paths):
    means, data = func(path)
    OSI_arr = np.zeros((3,25))
    for i in range(3):
        for j in range(25):
            OSI_arr[i,j] = OSI(data[i,:,j], angles)
    x_arr = np.arange(3)
    OSI_std = np.nanstd(OSI_arr, axis = 1)/np.sqrt(24)
    OSI_mean = np.nanmean(OSI_arr, axis = 1)
    FPS = data[:,0,:]
    FPS_mean = np.nanmean(FPS, axis =1 )
    FPS_std = np.nanstd(FPS, axis =1 )/np.sqrt(24)
    #  print((OSI_mean[0] - OSI_mean[1])/(np.sqrt(OSI_std[0]**2 + OSI_std[1]**2)))
    #  print((OSI_mean[0] - OSI_mean[2])/(np.sqrt(OSI_std[0]**2 + OSI_std[2]**2)))
    #  print(stats.norm.cdf(0,2.44)*2)
    #  print((OSI_mean[0] - OSI_mean[2] - OSI_std[2])/OSI_std[0])


    chi2_fit = Chi2Regression(alt_func, x_arr, OSI_mean, OSI_std)
    minuit = Minuit(chi2_fit, a = 1)
    minuit.errordef = 1
    minuit.migrad()
    print(stats.chi2.sf(minuit.fval,2), 'p add')


    #  ax2.scatter(x_arr, OSI_arr, ls = ls[idx], color = 'slategrey')
    ax.errorbar(x_arr, FPS_mean, yerr = FPS_std, ls = ls[idx], color = 'tab:red', label = labels[idx])
    ax2.errorbar(x_arr, OSI_mean, yerr = OSI_std, ls = ls[idx], color = 'slategrey', label = labels[idx])
    #  ax.scatter(x_arr, means[:,0], ls = ls[idx], color = 'tab:red')
    #  ax.errorbar(x_arr, means[:,0], ls = ls[idx], color = 'tab:red')



    #  for j in range(15):
    #      ax2.scatter(x_arr, OSI_arr[:,j], ls = ls[idx], color = 'slategrey')
    #      #  ax2.plot(x_arr, OSI_arr[:,j], ls = ls[idx], color = 'slategrey', label = labels[idx])
    #      ax.scatter(x_arr, data[:,0,j], ls = ls[idx], color = 'tab:red')
        #  ax.plot(x_arr, data[:0,j], ls = ls[idx], color = 'tab:red')
    #  ax2.scatter(x_arr, fwhm, ls = ls[idx], color = 'red')
    #  ax2.plot(x_arr,    fwhm, ls = ls[idx], color = 'red')
    #  print(means[:,0])
    #  print(idx)
ax2.set_ylim(0,1)
ax.set_ylim(5,30)
#  ax2.set_ylim(0,1)
#  ax.set_ylim(11,13.5)
ax.set_ylabel('Peak firing rate at $0^\circ$', color = 'tab:red')
ax2.set_ylabel('OSI', color = 'black')
ax2.set_xticks([0,1,2],['None', '$ Low \Delta E_K$', '$ High \Delta E_K$'] , rotation = 30)
ax2.legend(title = 'Ratio', loc = 4)
ax.text(-0.1, 1.05, 'D', fontweight = 'bold', transform = ax.transAxes, fontsize = 20)
#  ax.set_yticks(np.arange(11.5, 14, .5), np.arange(11.5, 13.5, .5))
#  ax2.set_yticks(np.arange(.6, .85, .05), np.round(np.arange(.6, .85, .05), 2))
fig.savefig('OSI', dpi =  200)
fig.savefig('FIG_4D.svg', dpi =  400)
fig.savefig('FIG_4D.pdf', dpi =  400)
plt.show()


   

