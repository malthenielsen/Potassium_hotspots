import numpy as np
from matplotlib import pyplot as plt
#  plt.style.use('science')
from matplotlib.gridspec import GridSpec

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
from glob import glob
import os
from scipy import stats
from scipy.signal import find_peaks
from iminuit import Minuit
import sys                                             # Module to see files and folders in directories

from ExternalFunctions import UnbinnedLH, BinnedLH, Chi2Regression
from ExternalFunctions import nice_string_output, add_text_to_ax   # Useful functions to print fit results on figure

def func(path):

    fnames_soma = glob(f'{path}/soma?*.npy')
    fnames_trunk = glob(f'{path}/stack?*.npy')
    fnames_trunk.sort(key=os.path.getctime)
    fnames_soma.sort(key=os.path.getctime)

    data = np.zeros((3,19,len(fnames_trunk)))


    for l, fname in enumerate(fnames_soma):
        print(fname)
        soma = np.load(fname)
        print(soma.shape)
        FPS_arr = np.zeros(19*3)
        v_arr = np.zeros(19*3)
        bin_ = []
        for i in range(len(soma)):
            inds, _ = find_peaks(soma[i], height = 0)
            FPS_arr[i]= len(inds)
        data[:,:,l] = FPS_arr.reshape(19,3).T


    mean = np.mean(data, axis = 2)
    eom = np.std(data, axis = 2)/np.sqrt(data.shape[2])
    return mean, eom


means3 = np.load('mean_0905.npy')
eoms3 = np.load( 'eom_0905.npy')
means2 = np.load('mean_0362.npy')
eoms2 = np.load( 'eom_0362.npy')
means1 = np.load('mean_20381.npy')
eoms1 = np.load('eom_20381.npy')
angles2 = np.arange(0,36,2)
angles1 = np.arange(20,38,1)
angles3 = np.arange(0,90,5)
angles = np.linspace(0,35,18)
angles3 = np.append(angles, np.array([40,50,75,90]))

mean = np.hstack([means3])#, means2,  means3])
eom = np.hstack([eoms3])#, eoms2, eoms3])
angles = np.hstack([angles3])#, angles2, angles3])
idx = np.argsort(angles)
angles = angles[idx]
mean = mean[:,idx]
eom = eom[:,idx]

data = np.vstack([angles, mean, eom])
print(data.shape)

#  np.save('neuron_data', data)
#  exit()


#  mean, eom = func('./3frac_5050_test')
#  print(mean.shape)
#  angles = np.arange(10,29,1)
#  print(angles.shape)

fig, ax = plt.subplots(1,1, figsize = (8,6))
for i in range(3):
    ax.errorbar(angles, mean[i,:], yerr = eom[i,:], ls = '')
    ax.scatter(angles, mean[i,:])

plt.show()

def chi_square(constant, y, y_err):
    return np.sum((y - constant)**2/y_err)

def lfunc(x, a):
    return  x*a

def altfunc(x, a):
    return x + a

def linfunc(x, a, b):
    return x*a + b

N = len(angles)

errors = np.zeros((2,N))
means = np.zeros((2,N))
errors_mul = np.zeros((2,N))
means_mul = np.zeros((2,N))

for j in range(2):
    for i in range(N):
        if mean[j,i] == 0:
            continue
        r_mean_0 = np.random.normal(mean[0,i], eom[0,i], 10000)
        r_mean_10 = np.random.normal(mean[j+1,i], eom[j+1,i], 10000)

        z = r_mean_10 - r_mean_0
        errors[j,i] = np.std(z)
        means[j,i] = np.mean(z)

        z = r_mean_10 / r_mean_0
        errors_mul[j,i] = np.std(z)
        means_mul[j,i] = np.mean(z)

#  eom[eom == 0] = 0.1

fig = plt.figure(figsize = (12,6))
gs = GridSpec(2,2)
axu = fig.add_subplot(gs[:,0])
#  axd = fig.add_subplot(gs[1,0])
axb = fig.add_subplot(gs[:,1])


#  axu.errorbar(angles, mean[0,:], yerr = eom[0,:], ls = 'dashed', marker = 'x', capsize = 2, color = 'teal')
#  axu.errorbar(angles, mean[1,:], yerr = eom[1,:], ls = 'dashed', marker ='x', capsize = 2, color = 'goldenrod'  )
axu.errorbar(angles, mean[2,:], yerr = eom[2,:], ls = '', marker = '.', capsize = 2, color = 'hotpink', label = 'High $\Delta E_K$')
#  axu.set_xlim(0,90)

#  ax[1].errorbar(angles, means[0,:], yerr = errors[0,:], ls = 'dashed', marker = 'o', capsize = 4, label = r'$\frac{Soma_{5} FR}{Soma_{0} FR}$' )
#  axd.errorbar(angles, means[1,:], yerr = errors[1,:], ls = 'dashed', marker = 'o', capsize = 4, label = r'$\frac{Soma_{5} FR}{Soma_{0} FR}$', color = 'red' )
#  ax2 = axd.twinx()
#  ax2.errorbar(angles, means_mul[0,:], yerr = errors_mul[0,:], ls = 'dashed', marker = 'o', capsize = 4, label = r'$\frac{Soma_{5} FR}{Soma_{0} FR}$' )
#  ax2.errorbar(angles, means_mul[1,:], yerr = errors_mul[1,:], ls = 'dashed', marker = 'o', capsize = 4, label = r'$\frac{Soma_{5} FR}{Soma_{0} FR}$' , color = 'black')
#  ax2.set_ylim(0,2)

#  axd.set( xlabel ='Angle from soma pref', ylabel = 'Differnce', title = '$\Delta E_{K}$ impact style', ylim = (0,3))
#  axd.yaxis.label.set_color('red')
#  ax2.set_ylabel('Fraction')
fig.suptitle('Comparison between Multiplicative \n and Addiative models')

fig1, ax1 = plt.subplots( figsize = (8,6))
#  axb.errorbar(mean[0,:], mean[1,:], yerr= eom[0,:], xerr = eom[1,:], ls ='')
axb.errorbar(mean[0,:], mean[2,:], yerr= eom[0,:], xerr = eom[2,:], ls ='', color = 'grey', alpha = .5)
axb.set_ylabel('angle from 0')
axu.set( ylabel = 'FPS')



for i in range(1,2):
    print(i)
    mask = eom[i, :] > 0
    chi2_fit = Chi2Regression(altfunc, mean[0,mask], mean[i,mask], eom[i,mask])
    minuit_chi2 = Minuit(chi2_fit, a = .1)
    minuit_chi2.errordef = 1
    minuit_chi2.migrad()
    axb.plot(mean[0,:], altfunc(mean[0,:], *minuit_chi2.values[:]), label = 'Fitted line', color = 'red')
    axu.plot(angles, altfunc(mean[0,:], *minuit_chi2.values[:]), label = 'Addiative', color = 'red', lw = 1)
    #  ax2.plot(angles, np.ones_like(angles)*minuit_chi2.values['a'], color = 'red')
    #  ax2.plot(angles, np.ones_like(angles)*minuit_chi2.values['a'], color = 'black')
    print(minuit_chi2.values[:], 'b')
    print(minuit_chi2.errors[:])
    print(stats.chi2.sf(minuit_chi2.fval, len(angles[mask]) - 1), 'p add')
    add_val = minuit_chi2.values['a']
    add_err = minuit_chi2.errors['a']


    #  chi2_fit = Chi2Regression(lfunc, angles[:10], mean[i,:10], errors[i,:10])
    chi2_fit = Chi2Regression(lfunc, mean[0,mask], mean[i,mask], eom[i,mask])
    minuit_chi2 = Minuit(chi2_fit, a = 1)
    minuit_chi2.errordef = 1
    minuit_chi2.migrad()
    #  ax1.plot(mean[0,:], lfunc(mean[0,:], *minuit_chi2.values[:]), label = 'Fitted line', color = 'red')
    #  ax[0].plot(angles, lfunc(mean[0,:], *minuit_chi2.values[:]), label = 'Fitted line', color = 'red')
    axb.plot(mean[0,:], lfunc(mean[0,:], *minuit_chi2.values[:]), label = 'Fitted line', color = 'black')
    axu.plot(angles, lfunc(mean[0,:], *minuit_chi2.values[:]), label = 'Multiplicative', color = 'black', lw = 1)
    #  ax2.plot(angles, np.ones_like(angles)*minuit_chi2.values['a'], color = 'black')
    print(minuit_chi2.values[:])
    print(minuit_chi2.errors[:])
    print(stats.chi2.sf(minuit_chi2.fval, len(angles[mask]) - 1), 'p mul')

    #  chi2_fit = Chi2Regression(linfunc, mean[0,mask], mean[i,mask], eom[i,mask])
    #  minuit_chi2 = Minuit(chi2_fit, a = 1, b = .1)
    #  minuit_chi2.errordef = 1
    #  minuit_chi2.migrad()
    #  ax1.plot(mean[0,:], linfunc(mean[0,:], *minuit_chi2.values[:]), label = 'Fitted line', color = 'red')
    #  ax[0].plot(angles, linfunc(mean[0,:], *minuit_chi2.values[:]), label = 'Multiplicative', color = 'red', lw = 2)
    #  ax2.plot(angles, np.ones_like(angles)*minuit_chi2.values['a'], color = 'red')
    #  print(minuit_chi2.values[:])
    #  print(minuit_chi2.errors[:])
    #  print(stats.chi2.sf(minuit_chi2.fval, len(angles[mask]) - 2), 'p mul')
d = {
        r'$\xi_{Mul}$ = ': [minuit_chi2.values['a'],minuit_chi2.errors['a']],
        r'$\xi_{Add}$ = ': [add_val, add_err],
        #  'Offset = ': [minuit_chi2.values['b'],minuit_chi2.errors['b']],
    }
text = nice_string_output(d, extra_spacing=2, decimals=3)
add_text_to_ax(0.4, 0.5, text, axu, fontsize=14)
axu.legend()

axb.set(xlabel = '$FPS_{Before}$', ylabel = '$FPS_{After}$')
axb.legend(['Addiative', 'Multiplicative', 'Data'])
axb.set_title('Fit to data')
axu.set_title('Fit overlaid on tuning curve')
#  axd.set_title('Isolated coeff from data')


fig.savefig('SI_FIG2.svg', dpi = 400)
fig.savefig('SI_FIG2.pdf', dpi = 400)

plt.show()


