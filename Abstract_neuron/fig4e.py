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

#  trunk_lengths = np.arange(100, 1500, 100)
#  trunk_lengths = np.arange(100, 1000, 50)
#  trunk_lengths = np.arange(50, 1000, 25)
trunk_lengths = np.arange(50, 500, 25)
trunk_lengths = np.append(trunk_lengths, np.array([750, 1000]))
#  trunk_lengths = np.arange(100, 500, 100)

#  trunk_lengths = np.append(trunk_lengths, np.array([2000, 2500, 3000, 4000]))
N_t = len(trunk_lengths)
print(N_t)
#  exit()
angles = np.linspace(0, 39, 18)
angles = np.append(angles,np.array([60,90]))
N_t = len(angles)

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
            #  inds, _ = find_peaks(trunk[i], height = -30)
            #  for idx in inds:
            #      trunk[i][idx - 55:idx + 55] = np.mean(trunk[i][idx+55:idx+60])

            if i%3 == 0:
                baseline = np.mean(trunk[i][3500:4000])
                #  fig, ax = plt.subplots(1,1, figsize = (7,6))
            trunk_tmp = trunk[i][5000:30000] - baseline
            #  plt.plot(trunk[i][:] - baseline)
            #  if i%3 == 2:
                #  plt.show()

            area = np.trapz(trunk_tmp, dx = 1/40)
            bin_.append(area)
        bin_ = np.array(bin_)
        data[:,:,l] = bin_.reshape(N_t,3).T


    mean = np.mean(data, axis = 2)
    eom = np.std(data,axis =2)/np.sqrt(data.shape[-1])
    return mean, eom 

#  paths = ['./new_tt/*.npy']#,
#paths = ['./trunk_test_scaled/current_large/folder_trunk_test/']#,
#  paths = ['./scaled/jan/']#,
#  paths = ['../gain/VM_tuning']#,
#  paths = ['./scaled/tuning_VM_17']#,
paths = ['./length_tuning/']
#  paths = ['./aoc4F/test/trace/']#,
        # './no_cluster/*.npy']
#  trunk_lenghts = np.arange(100,1000,100) + 300


mean, eom = func(paths[0])
#  mean_n, eom_n, t_test, residuals, T_large, R_val = func(paths[1])

fig, ax = plt.subplots(1,1, figsize = (7,7))
ax.legend(['No shift', 'Low shift', 'High shift'], title = '$\Delta E_K$')
ax.set(xlabel = 'Neuron lenghts[um]', ylabel = 'AOC [AU]', title = 'AOC as function of trunklenght')
#  plt.savefig('AOC_compar', dpi = 200)
#  plt.savefig('FIG_4G.svg', dpi = 400)
#  plt.savefig('FIG_4G.pdf', dpi = 400)
#  plt.show()


#  FR_mean = np.load('FR_mean.npy')
#  FR_eom = np.load('FR_eom.npy')

#  fig, ax = plt.subplots(1,1, figsize = (8,8))

colors = ['darkgrey', 'dodgerblue', 'goldenrod']
def func(x, a, b):
    #  return b*np.exp(a*(x-100))
    return b*x**-a

def bingo_bongo(x, a, b, c):
    #  return b*np.exp(a*(x-100))
    return b*x**-a + c

print(mean)

for i in range(3):
    #  mask = mean[i,:] > 0
    #  ax.errorbar(mean[i,:], FR_mean[i,:], yerr= FR_eom[i,:], xerr = eom[i,:], color = colors[i])
    ax.errorbar(angles, mean[i,:], eom[i,:], color = colors[i], ls = '-')


#  for i in range(3):
    #  p0, _ = curve_fit(func, trunk_lengths[:], mean[i,:], p0 = [1.29, 5000])
    #  fit_chi2_alt = Chi2Regression(bingo_bongo, trunk_lengths[:], mean[i,:], eom[i,:])
    #  minuit_chi2_alt = Minuit(fit_chi2_alt, a=p0[0], b=p0[1], c = 1000)
    #
    #  minuit_chi2_alt.errordef = 1.0
    #  minuit_chi2_alt.migrad()
    #  x_val = np.linspace(50, trunk_lengths[-1], 1000)
    #  ax.plot(x_val, bingo_bongo(x_val, *minuit_chi2_alt.values[:]), color = colors[i])
    #  ax.plot(x_val, bingo_bongo(x_val, *p0), color = colors[i])
    #  print(minuit_chi2_alt.values[:])
    #  print(1/2**(-minuit_chi2_alt.values[0]))
    #  print(minuit_chi2.errors[:])
    #  print(stats.chi2.sf(minuit_chi2_alt.fval, len(trunk_lengths) - 13), 'p')
    #  ax.set_ylim(0,200)
#  ax.legend(['No shift', 'Low shift', 'High shift'], title = '$\Delta E_K$')
#  ax.set(xlabel = 'Neuron lenghts[um]', ylabel = 'AOC [AU]', title = 'AOC as function of trunklenght')
#  plt.tight_layout()
#
#  fig.savefig('Fig4G.svg', dpi = 400)
#  fig.savefig('Fig4G.pdf', dpi = 400)
fig.savefig('FIG_4E.pdf', dpi = 200)

plt.show()







