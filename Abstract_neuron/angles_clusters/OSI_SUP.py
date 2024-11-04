import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
from glob import glob
import os
from scipy import stats, interpolate
from scipy.signal import find_peaks
from iminuit import Minuit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import sys                                             # Module to see files and folders in directories


def func(path):
    print(path)
    fnames_soma = glob(f'{path}/soma?*.npy')
    fnames_trunk = glob(f'{path}/stack?*.npy')
    fnames_trunk.sort(key=os.path.getctime)
    fnames_soma.sort(key=os.path.getctime)
    x_val = np.linspace(0,90,200)
    angles = np.array([0,5,10,15,20,30,40,50,80,89])



    data = np.zeros((3,22,len(fnames_soma)))
    print(len(fnames_soma))


    for l, fname in enumerate(fnames_soma):
        soma = np.load(fname)
        FPS_arr = np.zeros(22*3)
        for i in range(len(soma)):
            inds, _ = find_peaks(soma[i], height = 30)
            FPS_arr[i]= len(inds)
            #  if i%22 == 0:
                #  plt.plot(soma[i])
            
        data[:,:,l] = FPS_arr.reshape(22,3).T
    #  plt.show()

    mean = np.nanmean(data, axis = 2)
    #  plt.imshow(data[:,0,:])
    mean2 = (np.mean(data[:,0,:], axis = 1))
    print(data[:,0,:])
    #  plt.show()
    #  mean = data[:,:,0]
    fwhm = []
    #  for i in range(3):
    #      cs = interpolate.CubicSpline(angles, mean[i,:])
    #      yp = cs(x_val)
    #      idx = np.argmin(np.abs(yp - .5*np.max(yp)))
    #      fwhm.append(x_val[idx])
    #  plt.imshow(data[:,:,0], aspect = 'auto')
    #  plt.show()


    return mean, fwhm, mean2/mean2[0]

paths = ['./3frac_5050', './3frac_7525', './3frac_2575', './3frac_0100', './3frac_1000']
paths = ['./3frac_0100', './3frac_2575', './folder', './3frac_7525', './3frac_1000']
#  paths = ['./3frac_5050', './3frac_7525', './3frac_2575']
#  paths = ['./3frac_5050']
#  paths = ['./3frax_5050_test']

def OSI(arr, ori):
    ori = np.deg2rad(ori)
    top = np.sum(arr * np.exp(2*1j*ori))
    F = top/np.sum(arr)
    return 1  - np.arctan2(np.imag(F), np.real(F))
angles = np.linspace(0,35,18)
angles = np.append(angles, np.array([40,50,75,90]))

fig, ax = plt.subplots(1, 2, figsize = (10,6))
#  ax2 = ax.twinx()
#  color = ['teal', 'goldenrod', 'teal', '#75151E','#0047AB' ]
ls = ['solid', '--', '-.', ':', (0,(1,10))]
#  labels = ['50/50', '75/25', '25/75', '0/100', '100/0']
#  labels = ['0/', '75/25', '25/75', '0/100', '100/0']
for idx , path in enumerate(paths):
    means, fwhm, mean2 = func(path)
    #  print(means)
    OSI_arr = np.zeros(3)
    for i in range(3):
        OSI_arr[i] = OSI(means[i,:], angles)
    x_arr = np.arange(3)
    #  print(OSI_arr)
    ax[1].scatter(x_arr, OSI_arr, color = plt.cm.RdPu((idx + 1)/7))
    ax[1].plot(x_arr, OSI_arr, color = plt.cm.RdPu((idx + 1)/7))
    ax[0].scatter(x_arr, (mean2), color = plt.cm.RdPu((idx + 1)/7))
    ax[0].plot(x_arr, (mean2),color = plt.cm.RdPu((idx + 1)/7))
    #  ax[0].scatter(x_arr, np.mean(means[:,0:3], axis = 1), color = plt.cm.RdPu((idx + 1)/7))
    #  ax[0].plot(x_arr, np.mean(means[:,0:3], axis =1),color = plt.cm.RdPu((idx + 1)/7))
    #  ax2.scatter(x_arr, fwhm, ls = ls[idx], color = 'red')
    #  ax2.plot(x_arr,    fwhm, ls = ls[idx], color = 'red')
    #  print(means[:,0])
    #  print(idx)
ax[1].set_ylim(.8,.95)
#  ax[0].set_ylim(5,35)
ax[0].set_ylim(.8,3.5)
ax[0].set_ylabel('Peak firing rate at $0^\circ$', color = 'black')
ax[1].set_ylabel('OSI', color = 'black')
ax[1].set_xticks([0,1,2],['None', '$ Low \Delta E_K$', '$ High \Delta E_K$'] , rotation = 30)
ax[0].set_xticks([0,1,2],['None', '$ Low \Delta E_K$', '$ High \Delta E_K$'] , rotation = 30)
ax[1].legend(title = 'Ratio', loc = 4)
#  ax[0].set_yticks(np.arange(11.5, 14, .5), np.arange(11.5, 14, .5))
ax[1].set_yticks(np.arange(.8, 1, .05), np.round(np.arange(.8, 1, .05), 2))
cbaxes = inset_axes(ax[0], width="40%", height="5%")
plt.colorbar(plt.cm.ScalarMappable(norm = None, cmap = 'RdPu_r'), cax=cbaxes, ticks=[0,1], orientation='horizontal', label = 'Ratio of clustered segments')

#  cbaxes = inset_axes(ax[1], width="40%", height="5%")
#  plt.colorbar(plt.cm.ScalarMappable(norm = None, cmap = 'Greys'), cax=cbaxes, ticks=[0,1], orientation='horizontal', label = 'Ratio of clustered segments')
#  ax[0].text(-0.1, 1.05, 'D', fontweight = 'bold', transform = ax.transAxes, fontsize = 20)
fig.savefig('FIG_S3.svg', dpi =  400)
fig.savefig('FIG_S3.pdf', dpi =  400)

plt.show()


    
