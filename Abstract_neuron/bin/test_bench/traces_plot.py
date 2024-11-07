import numpy as np
from matplotlib import pyplot as plt
plt.style.use('science')
from glob import glob
import os
from scipy import stats
from scipy.signal import find_peaks
from iminuit import Minuit
import sys                                             # Module to see files and folders in directories

def func(path):
    #  fnames = glob('../longer_trunk/more_clusters/*.npy')
    fnames = glob(path)
    fnames.sort(key=os.path.getctime)

    fnames_soma = fnames[1::2]
    fnames_trunk = fnames[::2]



    data = np.zeros((2,20,len(fnames_trunk)))
    angles = np.arange(0,60,3)

    idx = [0, 9, 19, 29, 39]
    
    fig, ax = plt.subplots(5,1, figsize = (8,8), sharex = True, sharey = True)
    for l, fname in enumerate(fnames_soma):
        ax_idx = 0
        soma = np.load(fname)
        for i in range(soma.shape[0]):
            if i in idx:
                ax[ax_idx].plot(soma[i], alpha = .3)
                ax[ax_idx].set_xlim(20000,70000)
                ax[ax_idx].set_title(f'Stim angle {angles[i//2]}')
                ax_idx += 1
        if l == 10:
            fig.savefig('angle', dpi = 200)

            plt.show()
        #  exit()




#  path = './tuning/*.npy'
#  func(path)

def func(path):
    #  fnames = glob('../longer_trunk/more_clusters/*.npy')
    fnames = glob(path)
    fnames.sort(key=os.path.getctime)

    fnames_soma = fnames[1::2]
    fnames_trunk = fnames[::2]


    data = np.zeros((2,20,len(fnames_trunk)))
    angles = np.arange(0,60,3)

    idx = [0]
    
    fig, ax = plt.subplots(1,1, figsize = (8,8), sharex = True, sharey = True)
    for l, fname in enumerate(fnames_trunk):
        soma = np.load(fname)
        for i in range(soma.shape[0]):
            ax.plot(soma[i], alpha = 1)
            ax.set_xlim(20000,70000)
            ax.set_title(f'Stim angle {angles[i//2]}')
        plt.show()




path = './folder/*.npy'
func(path)

