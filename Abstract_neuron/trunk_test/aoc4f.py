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
from matplotlib import cm 
from matplotlib.colors import Normalize

plt.rcParams.update({'font.size': 14})
plt.rcParams['svg.fonttype'] = 'none'

cmap = plt.cm.cool
norm = Normalize(vmin=50, vmax=500)

# Create a ScalarMappable object
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

def rm(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')

#  trunk_lengths = np.arange(100, 1500, 100)
#  trunk_lengths = np.arange(100, 1000, 50)
trunk_lengths = np.arange(50, 1000, 25)
trunk_lengths = np.arange(50, 1000, 100)
trunk_lengths = np.arange(100, 500, 25)
#  trunk_lengths = np.arange(100, 500, 100)

#  trunk_lengths = np.append(trunk_lengths, np.array([2000, 2500, 3000, 4000]))
N_t = len(trunk_lengths)
print(N_t)
#  exit()

def func(path):
    fnames_trunk = glob(f'{path}/soma?*.npy')
    fnames_trunk.sort(key=os.path.getctime)


    colors = ['darkgrey', 'dodgerblue', 'goldenrod']

    data = np.zeros((3,N_t,len(fnames_trunk)))


    for l, fname in enumerate(fnames_trunk):
        if l != 12:
            continue
        trunk = np.load(fname)
        fig, ax = plt.subplots(3,1, figsize = (4,12), sharex = True)
        for i in range(len(trunk)):
            #  if i in [10,12,15,16]:
                #  continue
            ax[i%3].plot(trunk[i][6500:11000], color = plt.cm.cool_r(i/N_t))
            #  ax[i%3].plot(trunk[i][9000:14000], color = plt.cm.cool_r(i/N_t) )
            #  ax[i%3].plot(trunk[i][:], color = plt.cm.cool_r(i/N_t) )
            ax[i%3].set_xticks([])
            #  ax[i%3].set_yticks([])
        ax[0].legend()
        
        #  cbar = plt.colorbar(sm, ax=ax[1], orientation='vertical', pad=0.)
        #  cbar.set_label('Neuron length', rotation=270, labelpad=15)
        #  cbar.set_ticks([50, 500])
        ax[0].set_title('Low dEK')
        ax[1].set_title('Medium dEK')
        ax[2].set_title('High dEK')
        ax[2].set_xticks(np.linspace(0,5000,5), np.linspace(0,200,5))
        fig.savefig('4F.svg', dpi = 300)
        fig.savefig('4F.pdf', dpi = 300)
        plt.show()
        return None, None


#  paths = ['./aoc4F/test/trace/']#,
paths = ['./scaled/length_VM_AOC/']#,
#  paths = ['./scaled/tuning_VM_high/']#,
paths = ['./length_tuning/']#,


mean, eom = func(paths[0])







