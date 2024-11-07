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
trunk_lengths = np.arange(50, 1000, 25)
trunk_lengths = np.arange(50, 250, 25)

#  trunk_lengths = np.append(trunk_lengths, np.array([2000, 2500, 3000, 4000]))
N_t = len(trunk_lengths)
print(N_t)
#  exit()

def func(path):
    #  fnames = glob(path)
    #  fnames.sort(key=os.path.getctime)
    #  fnames = fnames[:5]

    #  fnames_soma = fnames[1::2]
    #  fnames_trunk = fnames
    fnames_trunk = glob(f'{path}/soma?*.npy')
    fnames_trunk.sort(key=os.path.getctime)


    colors = ['darkgrey', 'dodgerblue', 'goldenrod']

    data = np.zeros((3,N_t,len(fnames_trunk)))


    for l, fname in enumerate(fnames_trunk):
        trunk = np.load(fname)
        bin_ = []
        
        for i in range(len(trunk)):
            #  inds, _ = find_peaks(trunk[i], height = -30)
            #  print(len(trunk[i]))
            #  for idx in inds:
            #      trunk[i][idx - 55:idx + 55] = np.mean(trunk[i][idx+55:idx+60])

            if i%3 == 0:
                #  baseline = np.mean(trunk[i][2500:3000])
                #  fig, ax = plt.subplots(1,1, figsize = (7,6))
            #  trunk_tmp = trunk[i][5000:20000] - baseline
            #  plt.plot(trunk[i][:])# - baseline)
            #  if i%3 == 2:
                #  plt.show()

            #  area = np.trapz(trunk_tmp, dx = .001)
            inds, _ = find_peaks(trunk[i], height = 0)
            bin_.append(len(inds))
        bin_ = np.array(bin_)
        data[:,:,l] = bin_.reshape(N_t,3).T


    mean = np.mean(data, axis = 2)
    eom = np.std(data,axis =2)/np.sqrt(data.shape[-1])
    return mean, eom 

#  paths = ['./new_tt/*.npy']#,
#  paths = ['./trunk_test_scaled/current_30/folder_trunk_test/']#,
#  paths = ['./trunk_test_scaled/current_30']#,
paths = ['./scaled/spikes/jan']#,
        # './no_cluster/*.npy']
#  trunk_lenghts = np.arange(100,1000,100) + 300


mean, eom = func(paths[0])
#  mean_n, eom_n, t_test, residuals, T_large, R_val = func(paths[1])

fig, ax = plt.subplots(1,1, figsize = (7,6))
ax.text(-0.1, 1.05, '4G', fontweight = 'bold', transform = ax.transAxes, fontsize = 20)
def func(x,a,b):
    return a*x + b

colors = ['darkgrey', 'dodgerblue', 'goldenrod']

#  for i in range(3):
#      fit_chi2 = Chi2Regression(func, trunk_lengths[:], mean[i,:], eom[i,:])
#      minuit_chi2 = Minuit(fit_chi2, a=0.0, b=0.0)
#      minuit_chi2.errordef = 1.0
#      minuit_chi2.migrad()
#      ax.plot(trunk_lengths, func(trunk_lengths, *minuit_chi2.values[:]), color = colors[i])
#      print(minuit_chi2.values[:])
#      print(minuit_chi2.errors[:])
#      print(stats.chi2.sf(minuit_chi2.fval, len(trunk_lengths) - 2))

#  for i in range(3):
    #  fit_chi2 = Chi2Regression(func, trunk_lenghts, mean_n[i,:], eom_n[i:])
    #  minuit_chi2 = Minuit(fit_chi2, a=0.0, b=0.0)
    #  minuit_chi2.errordef = 1.0
    #  minuit_chi2.migrad()
    #  ax.plot(trunk_lenghts, func(trunk_lenghts, *minuit_chi2.values[:]), color = colors[i], ls = '--')
    #  print(minuit_chi2.values[:])
    #  print(minuit_chi2.errors[:])



ax.errorbar(trunk_lengths, mean[0,:], eom[0,:], ls = '', capsize = 0, color = colors[0], marker = '')
ax.errorbar(trunk_lengths, mean[1,:], eom[1,:], ls = '', capsize = 0, color = colors[1], marker = '')
ax.errorbar(trunk_lengths, mean[2,:], eom[2,:], ls = '', capsize = 0, color = colors[2], marker = '')
#  ax.set_xticks([700, 1200, 1700, 2200, 2700, 3200, 3700],[700, 1200, 1700, 2200, 2700, 3200, 3700])

#  ax.errorbar(trunk_lenghts, mean_n[0,:], eom_n[0,:], ls = '', capsize = 4, color = colors[0], marker = 'x')
#  ax.errorbar(trunk_lenghts, mean_n[1,:], eom_n[1,:], ls = '', capsize = 4, color = colors[1], marker = 'x')
#  ax.errorbar(trunk_lenghts, mean_n[2,:], eom_n[2,:], ls = '', capsize = 4, color = colors[2], marker = 'x')
ax.legend(['No shift', 'Low shift', 'High shift'], title = '$\Delta E_K$')
ax.set(xlabel = 'Neuron lenghts[um]', ylabel = 'AOC [AU]', title = 'AOC as function of neuron lenght')
#  plt.savefig('AOC_compar', dpi = 200)
#  plt.savefig('FIG_4G.svg', dpi = 400)
#  plt.savefig('FIG_4G.pdf', dpi = 400)
#  plt.show()


FR_mean = np.load('FR_mean.npy')
FR_eom = np.load('FR_eom.npy')

#  fig, ax = plt.subplots(1,1, figsize = (8,8))

def func(x, a, b):
    #  return b*np.exp(a*(x-100))
    return b*x**-a

def bingo_bongo(x, a, b):
    #  return b*np.exp(a*(x-100))
    return b*x**-a

for i in range(3):
    #  mask = mean[i,:] > 0
    #  ax.errorbar(mean[i,:], FR_mean[i,:], yerr= FR_eom[i,:], xerr = eom[i,:], color = colors[i])
    ax.errorbar(trunk_lengths[2:], mean[i,2:], eom[i,2:], color = colors[i], ls = '')

    #  p0, _ = curve_fit(func, trunk_lengths[2:], mean[i,2:], p0 = [1.29, 5000])
    #  #  print(p0)
    #  fit_chi2_alt = Chi2Regression(bingo_bongo, trunk_lengths[2:], mean[i,2:], eom[i,2:])
    #  #  minuit_chi2 = Minuit(fit_chi2, a=0.005, b=mean[i,0])
    #  minuit_chi2_alt = Minuit(fit_chi2_alt, a=p0[0], b=p0[1])
    #  #  minuit_chi2_alt.fixed['a'] = True
    #  #  minuit_chi2_alt.fixed['b'] = True
    #  #
    #  minuit_chi2_alt.errordef = 1.0
    #  minuit_chi2_alt.migrad()
    #  x_val = np.linspace(100, 1100, 1000)
    #  ax.plot(x_val, bingo_bongo(x_val, *minuit_chi2_alt.values[:]), color = colors[i])
    #  #  ax.plot(x_val, bingo_bongo(x_val, *p0), color = colors[i])
    #  print(minuit_chi2_alt.values[:])
    #  print(1/2**(-minuit_chi2_alt.values[0]))
    #  #  print(minuit_chi2.errors[:])
    #  print(stats.chi2.sf(minuit_chi2_alt.fval, len(trunk_lengths) - 3), 'p')
    #  ax.set_ylim(0,200)

#  fig.savefig('Fig4G.svg', dpi = 400)

plt.show()







