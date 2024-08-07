import numpy as np
from matplotlib import pyplot as plt

plt.style.use('K_PAPER')
from glob import glob
import os
from scipy import stats
from scipy.signal import find_peaks
from iminuit import Minuit
import sys                                             # Module to see files and folders in directories
from appstatpy.ExternalFunctions import *

#  from ExternalFunctions import UnbinnedLH, BinnedLH, Chi2Regression
#  from ExternalFunctions import nice_string_output, add_text_to_ax   # Useful functions to print fit results on figure

print(len(np.arange(0,90,4)))
def func(path):
    #  fnames = glob('../longer_trunk/more_clusters/*.npy')
    #  fnames = glob(path)
    #  fnames.sort(key=os.path.getctime)

    fnames_soma = glob(f'{path}/soma?*.npy')
    fnames_trunk = glob(f'{path}/stack?*.npy')
    fnames_trunk.sort(key=os.path.getctime)
    fnames_soma.sort(key=os.path.getctime)
    #  fnames_trunk = fnames[::2]
    #  print(fnames_soma)

    vinkel = np.arange(0,90,4)



    #  data = np.zeros((3,23,len(fnames_trunk)))
    #  data = np.zeros((3,10,len(fnames_trunk)))
    #  dataV = np.zeros((3,10,len(fnames_trunk)))
    data = np.zeros((3,22,len(fnames_soma)))
    dataV = np.zeros((3,22,len(fnames_soma)))
    dataI = np.zeros((22,3,len(fnames_soma), 100))


    for l, fname in enumerate(fnames_soma):
        print(fname)
        soma = np.load(fname)
        FPS_arr = np.zeros(22*3)
        v_arr = np.zeros(22*3)
        ISD = np.zeros((22*3, 100))

        bin_ = []
        binV = []
        #  print(len(soma), 'soma_length')
        #  print(fname)

        for i in range(len(soma)):
            inds, _ = find_peaks(soma[i], height = 50)
            FPS_arr[i]= len(inds)
            if len(inds) > 1:
                isd = np.diff(inds)
                ISD[i,:len(isd)] = isd

            v_arr[i] = vinkel[i%22]
            #  plt.plot(soma[i])
            #  print(i)
        #  plt.show()
        #  exit()
            #  bin_.append(len(inds))
            #  binV.append(vinkel[i%23])



        #  data[:,:,l] = np.array(bin_).reshape(23,3).T
        data[:,:,l] = FPS_arr.reshape(22,3).T
        dataV[:,:,l] = v_arr.reshape(3,22)
        #  dataV[:,:,l] = np.array(binV).reshape(23,3).T
        #  print(dataV[:,:,l])
        dataI[:,:,l,:] = ISD.reshape(22,3,100)

    t_test = np.zeros((2,22))
    residuals_arr = np.zeros((2,22,len(fnames_soma)))
    T_large = np.zeros(2)
    R_val = np.zeros(2)


    for i in range(22):
        for j in range(2):
            residuals = np.diff(data[[0,j+1], i, :], axis = 0).ravel() 
            residuals_arr[j,i,:] = residuals
            sdom = np.std(residuals, ddof = 1)/np.sqrt(len(residuals))
            t = np.mean(residuals) / sdom
            #  print(stats.t.sf(t, len(residuals)-1))
            t_test[j,i] = stats.t.sf(t, len(residuals)-1)
    for i in range(2):
        residuals = residuals_arr[i,:,:].ravel()
        sdom = np.std(residuals, ddof = 1)/np.sqrt(len(residuals))
        t = np.mean(residuals) / sdom
        T_large[i] = stats.t.sf(t, len(residuals)-1)
        R_val[0] = t
        R_val[1] = len(residuals)-1

    #  print(T_large)

    mean = np.mean(data, axis = 2)
    print(data)
    mean_v = np.mean(dataV, axis = 2)
    #  print(dataV[:,:,0])
    #  print(mean_v, 'vinkel')
    #  print(mean_v.ravel().reshape(3,23))
    eom = np.std(data, axis = 2)/np.sqrt(data.shape[2])
    print(mean.T)
    np.save('fit/mean_0905', mean)
    np.save('fit/eom_0905', eom)
    return mean, eom, T_large, R_val, residuals_arr, dataI

#  paths = ['./tuning4_frac_rune']
#  paths = ['./3frac_domcluster']
#  paths = ['./3frax_5050_test/3frac_5050']
#  paths = ['./3frac_5050']

paths = ['./folder']


def get_traces(path):
    #  fnames = glob(path)
    #  fnames.sort(key=os.path.getctime)
    fnames_soma = glob(f'{path}/soma?*.npy')
    fnames_trunk = glob(f'{path}/stack?*.npy')

    #  fnames_soma = fnames[1::2]
    #  fnames_trunk = fnames[::2]
    inds = [0]
    return_list = []
    for l, fname in enumerate(fnames_soma):
        if l in inds:
            return_list.append(np.load(fname))
    return return_list


def gamma(x, N, a, b):
    return N * stats.gamma.pdf(x, a, scale = 1/b) 


#  paths = ['../longer_trunk/many_runs/*.npy']
for path in paths:
    means, eoms, T_val, R_val, residuals, ISD = func(path)

#  print(ISD.shape)
#  ISD[ISD == 0] = np.nan
#  for i in range(6):
#      fig, ax = plt.subplots(1,1, figsize = (8,6))
#      for j in range(3):
#          ax.hist(ISD[i,j,:, :].ravel()*0.0001, bins = np.linspace(0, .20, 40), histtype = 'step')
#          freq, bins = np.histogram(ISD[i,j,:, :].ravel()*0.0001, bins = np.linspace(0, .20, 40))
#          bins = (bins[1:] + bins[:-1])/2
#          freq_err = np.sqrt(freq)
#          mask = freq > 0
#          chi2_fit = Chi2Regression(gamma, bins[mask], freq[mask], freq_err[mask])
#          gamma_fit = Minuit(chi2_fit, N = 1, a = 1, b = 1)
#          gamma_fit.errordef = 1
#          gamma_fit.migrad()
#          ax.plot(bins, gamma(bins, *gamma_fit.values[:]))
#
#          data = ISD[i,j,:,:].ravel()
#          data = np.nan_to_num(data, 0)
#          N = np.sum(np.where(data == 0, 1, 0))
#          #  print(N)
#
#          print(np.std(data)/np.sqrt(N))
#
#          #  print(np.nanstd(ISD[i,j,:,:].ravel()))
#
#
#
#      plt.show()
#
#  exit()








def OSI(arr, ori):
    ori = np.deg2rad(ori)
    top = np.sum(arr * np.exp(2*1j*ori))
    F = top/np.sum(arr)
    return 1 - np.arctan2(np.imag(F), np.real(F))

#  angles = np.array([0,5,10,15,20,30,40,50,80,89])
#  angles = np.arange(0,90,5)
angles = np.linspace(0,35,18)
angles = np.append(angles, np.array([40,50,75,90]))

#  angles = np.arange(10,28,1)

print(OSI(means[0,:], angles))
print(OSI(means[1,:], angles))
print(OSI(means[2,:], angles))





def chi_square(constant, y, y_err):
    return np.sum((y - constant)**2/y_err)

def lfunc(x, a):
    return  x*a

def altfunc(x, a):
    return x + a


#  angles = np.arange(0,90,4).astype(float)
#  angles = np.array([0,5,10,15,20,30,40,50,80,89])
#  angles = np.arange(0,90,5)
#  angles = np.linspace(10,30,18)

fig_residuals, ax_res = plt.subplots(1,1, figsize = (8,6))
#  residuals
ax_res.hlines(0, 0, 90, ls = 'dashed', color = 'black', alpha = .7)
ax_res.errorbar(angles, np.mean(residuals[0,:,:], axis = 1),np.std(residuals[0,:,:], axis = 1)/np.sqrt(residuals.shape[2]), color = 'hotpink', ls = '', marker = 'o', capsize = 4, label  = r'Low $\Delta EK$')
ax_res.errorbar(angles, np.mean(residuals[1,:,:], axis = 1),np.std(residuals[1,:,:], axis = 1)/np.sqrt(residuals.shape[2]), color = 'teal', ls = '', marker = 'o', capsize = 4, label = r'High $\Delta EK$')
ax_res.legend(title = '$\Delta E_K$')
ax_res.set_ylim(-1, 2)
ax_res.set(ylabel = '$\Delta$ firing', xlabel = r'$\Delta\theta_{soma prefered}$', title = 'Residual change, caused by k_ext')
ax_res.text(40, -.7, f'Paired one tailed Students T test \n P-value \n 5mV = {T_val[0]} \n 10mV = {T_val[1]}', size = 13)
fig_residuals.savefig('residuals', dpi = 200)

errors = np.zeros((2,22))
mean = np.zeros((2,22))
errors_mul = np.zeros((2,22))
mean_mul = np.zeros((2,22))

#  mean_0 = means[0,:]
#  mean_10 = means[1,:]
#  eom_0 = eoms[0,:]
#  eom_10 = eoms[1,:]
for j in range(2):
    for i in range(22):
        if means[j,i] == 0:
            continue
        r_mean_0 = np.random.normal(means[0,i], eoms[0,i], 10000)
        r_mean_10 = np.random.normal(means[j+1,i], eoms[j+1,i], 10000)
        z = r_mean_10 - r_mean_0
        errors[j,i] = np.std(z)
        mean[j,i] = np.mean(z)
        z = r_mean_10 / r_mean_0
        errors_mul[j,i] = np.std(z)
        mean_mul[j,i] = np.mean(z)


from scipy.optimize import minimize

fig, ax = plt.subplots(2,1, figsize = (8,8))
#  ax.set_ylim(.5, 2)

ax[0].errorbar(angles, means[0,:], yerr = eoms[0,:], ls = 'dashed', marker = 'o', capsize = 4, label = r'$\frac{Soma_{5} FR}{Soma_{0} FR}$' )
ax[0].errorbar(angles, means[1,:], yerr = eoms[1,:], ls = 'dashed', marker = 'o', capsize = 4, label = r'$\frac{Soma_{5} FR}{Soma_{0} FR}$' )
ax[0].errorbar(angles, means[2,:], yerr = eoms[2,:], ls = 'dashed', marker = 'o', capsize = 4, label = r'$\frac{Soma_{10} FR}{Soma_{0} FR}$' )
ax[1].errorbar(angles, mean[0,:], yerr = errors[0,:], ls = 'dashed', marker = 'o', capsize = 4, label = r'$\frac{Soma_{5} FR}{Soma_{0} FR}$' )
ax[1].errorbar(angles, mean[1,:], yerr = errors[1,:], ls = 'dashed', marker = 'o', capsize = 4, label = r'$\frac{Soma_{5} FR}{Soma_{0} FR}$' )
ax2 = ax[1].twinx()
ax2.errorbar(angles, mean_mul[0,:], yerr = errors_mul[0,:], ls = 'dashed', marker = 'o', capsize = 4, label = r'$\frac{Soma_{5} FR}{Soma_{0} FR}$' )
ax2.errorbar(angles, mean_mul[1,:], yerr = errors_mul[1,:], ls = 'dashed', marker = 'o', capsize = 4, label = r'$\frac{Soma_{5} FR}{Soma_{0} FR}$' )
ax2.set_ylim(0,2)
ax[1].set(xlabel ='Angle from soma pref', ylabel = 'Fraction', title = '$\Delta E_{K}$ impact style', ylim = (0,3))

fig1, ax1 = plt.subplots( figsize = (8,6))
ax1.errorbar(means[0,:], means[1,:], yerr= eoms[0,:], xerr = eoms[1,:], ls ='')
ax1.errorbar(means[0,:], means[2,:], yerr= eoms[0,:], xerr = eoms[2,:], ls ='')


#  eoms[eoms == 0] = .2


for i in range(1,3):
    print(i)
    mask = eoms[i, :] > 0
    chi2_fit = Chi2Regression(altfunc, means[0,mask], means[i,mask], eoms[i,mask])
    minuit_chi2 = Minuit(chi2_fit, a = .1)
    minuit_chi2.errordef = 1
    minuit_chi2.migrad()
    ax1.plot(means[0,:], altfunc(means[0,:], *minuit_chi2.values[:]), label = 'Fitted line', color = 'black')
    ax[0].plot(angles, altfunc(means[0,:], *minuit_chi2.values[:]), label = 'Fitted line', color = 'black')
    print(minuit_chi2.values[:], 'b')
    print(minuit_chi2.errors[:])
    print(stats.chi2.sf(minuit_chi2.fval, len(angles[mask]) - 1), 'p add')


    #  chi2_fit = Chi2Regression(lfunc, angles[:10], mean[i,:10], errors[i,:10])
    chi2_fit = Chi2Regression(lfunc, means[0,mask], means[i,mask], eoms[i,mask])
    minuit_chi2 = Minuit(chi2_fit, a = 1)
    minuit_chi2.errordef = 1
    minuit_chi2.migrad()
    ax1.plot(means[0,:], lfunc(means[0,:], *minuit_chi2.values[:]), label = 'Fitted line', color = 'red')
    ax[0].plot(angles, lfunc(means[0,:], *minuit_chi2.values[:]), label = 'Fitted line', color = 'red')
    print(minuit_chi2.values[:])
    print(minuit_chi2.errors[:])
    print(stats.chi2.sf(minuit_chi2.fval, len(angles[mask]) - 1), 'p mul')
d = {
        'Slope = ': [minuit_chi2.values['a'],minuit_chi2.errors['a']],
        #  'Offset = ': [minuit_chi2.values['b'],minuit_chi2.errors['b']],
    }
text = nice_string_output(d, extra_spacing=2, decimals=3)
add_text_to_ax(0.1, 0.7, text, ax[0], fontsize=14)

import matplotlib.gridspec as gridspec
fig = plt.figure(tight_layout = True, figsize = (7,6))
gs = gridspec.GridSpec(3,4)
ax_tune = fig.add_subplot(gs[:,:])

def gauss(x, a, b, c):
    return a * np.exp(-(x - b)**2/(c**2))

colors = ['darkgrey', 'dodgerblue', 'goldenrod']

#  angles = np.linspace(0,38,18)
for i in range(3):
    mask = eoms[i, :] > 0
    chi2_fit = Chi2Regression(gauss, angles[mask], means[i,mask], eoms[i,mask])
    minuit_chi2 = Minuit(chi2_fit, a = 1, b = 0, c = 1)
    minuit_chi2.fixed['b'] = True
    minuit_chi2.errordef = 1
    minuit_chi2.migrad()
    #  ax_tune.plot(np.arange(0,90,.1), gauss(np.arange(0,90,.1), *minuit_chi2.values[:]), color = colors[i])
    print(minuit_chi2.values[:], 'gauss')
    print(minuit_chi2.errors[:])
    print(stats.chi2.sf(minuit_chi2.fval, len(angles[mask]) - 3), 'gauss fit')


ax_tune.errorbar(angles,  means[0,:], eoms[0,:], ls = 'solid', marker = '', capsize = 0, label = 'No shift', color = 'darkgrey')
ax_tune.errorbar(angles,  means[1,:], eoms[1,:], ls = 'solid', marker = '', capsize = 0, label = 'Low shift', color = 'dodgerblue')
ax_tune.errorbar(angles,  means[2,:], eoms[2,:], ls = 'solid', marker = '', capsize = 0, label = 'High shift', color = 'goldenrod')
ax_tune.legend(title = '$\Delta E_K$')
ax_tune.set(ylabel = 'Firing rate', xlabel = r'$\Delta\theta_{soma prefered}$', title = 'Soma firingrate')
ax_tune.text(-0.1, 1.05, 'C', fontweight = 'bold', transform = ax_tune.transAxes, fontsize = 20)
#  plt.savefig('Tuning_quick', dpi = 200)
plt.savefig('FIG_4C.svg', dpi = 400)
plt.savefig('FIG_4C.pdf', dpi = 400)

fig = plt.figure(tight_layout = True, figsize = (7,6))
gs = gridspec.GridSpec(3,15)
ax_tune = fig.add_subplot(gs[:,:10])
ax_tune_late = fig.add_subplot(gs[:,11:], sharey = ax_tune)
ax_tune_late.spines['left'].set_visible(False)
#  ax_tune_late.set_yticks([])

ax_tune.errorbar(angles[:13],  means[0,:13], eoms[0,:13], ls = 'solid', marker = '', capsize = 0, label = 'No shift', color = 'darkgrey')
ax_tune.errorbar(angles[:13],  means[1,:13], eoms[1,:13], ls = 'solid', marker = '', capsize = 0, label = 'Low shift', color = 'dodgerblue')
ax_tune.errorbar(angles[:13],  means[2,:13], eoms[2,:13], ls = 'solid', marker = '', capsize = 0, label = 'High shift', color = 'goldenrod')
ax_tune.set_xlim(0,25)

ax_tune_late.errorbar(angles[20:],  means[0,20:], eoms[0,20:], ls = 'solid', marker = '', capsize = 0, label = 'No shift', color = 'darkgrey')
ax_tune_late.errorbar(angles[20:],  means[1,20:], eoms[1,20:], ls = 'solid', marker = '', capsize = 0, label = 'Low shift', color = 'dodgerblue')
ax_tune_late.errorbar(angles[20:],  means[2,20:], eoms[2,20:], ls = 'solid', marker = '', capsize = 0, label = 'High shift', color = 'goldenrod')
ax_tune_late.set_xlim(80,90)
ax_tune_late.set_xticks([80,85,90], [80,85,90])

ax_tune.legend(title = '$\Delta E_K$')
ax_tune.set(ylabel = 'Firing rate', xlabel = r'$\Delta\theta_{soma prefered}$', title = 'Soma firingrate')
ax_tune.text(-0.1, 1.05, 'C', fontweight = 'bold', transform = ax_tune.transAxes, fontsize = 20)
plt.savefig('FIG_4C_split.svg', dpi = 400)
plt.savefig('FIG_4C_split.pdf', dpi = 400)
plt.show()

exit()





idx = [0, 1, 2]
traces = get_traces(path)
traces = traces[0].reshape(22,3,-1)
timepoints = np.arange(0000,30000,1)
ones = np.ones(30000)*-70
fig, ax = plt.subplots(5,3, figsize = (10,12), sharex = False, sharey = True)
#  [axi.set_axis_off() for axi in ax.ravel()]
for axi in ax.ravel():
    axi.spines['top'].set_visible(False)
    #  axi.spines['bottom'].set_visible(False)
    #  axi.spines['left'].set_visible(False)
    axi.spines['right'].set_visible(False)
    #  axi.set_yticks([])
    #  axi.set_xticks([])

#  ax[0,0].text(-0.1, 1.15, 'B', fontweight = 'bold', transform = ax[0,0].transAxes, fontsize = 20)
ax[0,0].plot(traces[0, 0, :], color = 'darkgrey',  lw=1)
ax[0,1].plot(traces[0, 1, :], color = 'dodgerblue',  lw=1)
ax[0,2].plot(traces[0, 2, :], color = 'goldenrod',  lw=1)

#  inds, _ = find_peaks(traces[0,0,5000:30000], height = -30)
#  ax[0,0].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)
#  inds, _ = find_peaks(traces[0,1,5000:30000], height = -30)
#  ax[0,1].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)
#  inds, _ = find_peaks(traces[0,2,5000:30000], height = -30)
#  ax[0,2].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)

ax[1,0].plot(traces[3, 0, :], color = 'darkgrey',  lw=1)
ax[1,1].plot(traces[3, 1, :], color = 'dodgerblue',  lw=1)
ax[1,2].plot(traces[3, 2, :], color = 'goldenrod',  lw=1)
#  inds, _ = find_peaks(traces[3,0,5000:30000], height = -30)
#  ax[1,0].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)
#  inds, _ = find_peaks(traces[3,1,5000:30000], height = -30)
#  ax[1,1].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)
#  inds, _ = find_peaks(traces[3,2,5000:30000], height = -30)
#  ax[1,2].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)

ax[2,0].plot(traces[9, 0, :], color = 'darkgrey',  lw=1)
ax[2,1].plot(traces[9, 1, :], color = 'dodgerblue',  lw=1)
ax[2,2].plot(traces[9, 2, :], color = 'goldenrod',  lw=1)
#  inds, _ = find_peaks(traces[9,0,5000:30000], height = -30)
#  ax[2,0].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)
#  inds, _ = find_peaks(traces[9,1,5000:30000], height = -30)
#  ax[2,1].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)
#  inds, _ = find_peaks(traces[9,2,5000:30000], height = -30)
#  ax[2,2].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)

ax[3,0].plot(traces[12, 0, :], color = 'darkgrey',  lw=1)
ax[3,1].plot(traces[12, 1, :], color = 'dodgerblue',  lw=1)
ax[3,2].plot(traces[12, 2, :], color = 'goldenrod',  lw=1)
#  inds, _ = find_peaks(traces[12,0,5000:30000], height = -30)
#  ax[3,0].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)
#  inds, _ = find_peaks(traces[12,1,5000:30000], height = -30)
#  ax[3,1].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)
#  inds, _ = find_peaks(traces[12,2,5000:30000], height = -30)
#  ax[3,2].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)

ax[4,0].plot(traces[-1, 0, :], color = 'darkgrey',  lw=1)
ax[4,1].plot(traces[-1, 1, :], color = 'dodgerblue',  lw=1)
ax[4,2].plot(traces[-1, 2, :], color = 'goldenrod',  lw=1)
#  inds, _ = find_peaks(traces[-1,0,20000:50000], height = -30)
#  ax[4,0].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)
#  inds, _ = find_peaks(traces[-1,1,20000:50000], height = -30)
#  ax[4,1].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)
#  inds, _ = find_peaks(traces[-1,2,20000:50000], height = -30)
#  ax[4,2].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)

ax[0,0].set_ylabel('$0^\circ$', rotation = 'horizontal')
ax[1,0].set_ylabel('$22^\circ$', rotation = 'horizontal')
ax[2,0].set_ylabel('$45^\circ$', rotation = 'horizontal')
ax[3,0].set_ylabel('$67^\circ$', rotation = 'horizontal')
ax[4,0].set_ylabel('$90^\circ$', rotation = 'horizontal')
ax[4,1].set_xlabel('Time[ms]')
#  ax[4,0].set_xticks(timepoints, time)
#  ax[4,1].set_xticks(timepoints, time)
#  ax[4,2].set_xticks(timepoints, time)
ax[0,0].set_title('No ionic shift')
ax[0,1].set_title('$\Delta E_K$ low')
ax[0,2].set_title('$\Delta E_K$ high')
#  plt.tight_layout()
#  ax[4].legend(title = '$\Delta E_K$')
fig.savefig('Orientation_tuning', dpi = 200)
fig.savefig('FIG_4B_shade_with_axes.svg', dpi = 400)
fig.savefig('FIG_4B_shade_with_axes.pdf', dpi = 400)

plt.show()

#  idx = [0, 1, 2]
#  traces = get_traces(path)
#  traces = traces[0].reshape(18,3,-1)
#  for i in range(18):
#      fig, ax = plt.subplots(3, 1, figsize = (8,8))
#      for j in range(3):
#          ax[j].plot(traces[i,j,:])
#      fig.suptitle(i)
#      plt.show()




timepoints = np.arange(0, len(traces[3,0,:]),1)
ones = np.ones_like(timepoints)*-70
fig, ax = plt.subplots(5,1, figsize = (6,12), sharex = False, sharey = True)
#  [axi.set_axis_off() for axi in ax.ravel()]
for axi in ax.ravel():
    axi.spines['top'].set_visible(False)
    axi.spines['bottom'].set_visible(False)
    axi.spines['left'].set_visible(False)
    axi.spines['right'].set_visible(False)
    axi.set_yticks([])
    axi.set_xticks([])

#  ax[0].text(-0.1, 1.15, 'B', fontweight = 'bold', transform = ax[0,0].transAxes, fontsize = 20)
ax[0].plot(traces[0, 2, :], color = 'darkgrey',  lw=1)
#  ax[0].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)
#  inds, _ = find_peaks(traces[0,1,:], height = -30)

ax[1].plot(traces[3, 2, :], color = 'darkgrey',  lw=1)
#  inds, _ = find_peaks(traces[3,0,:], height = -30)
#  ax[1].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)

ax[2].plot(traces[9, 2, :], color = 'darkgrey',  lw=1)
#  inds, _ = find_peaks(traces[9,0,:], height = -30)
#  ax[2].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)

ax[3].plot(traces[12, 2, :], color = 'darkgrey',  lw=1)
#  inds, _ = find_peaks(traces[12,0,:], height = -30)
#  ax[3].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)

ax[4].plot(traces[-1, 2, :], color = 'darkgrey',  lw=1)
#  inds, _ = find_peaks(traces[-1,0,:], height = -30)
#  ax[4].scatter(timepoints[inds], ones[inds], marker = '|', color = 'black', s = 5)
#  inds, _ = find_peaks(traces[-1,1,:], height = -30)

ax[0].set_ylabel('$0^\circ$', rotation = 'horizontal')
ax[1].set_ylabel('$22^\circ$', rotation = 'horizontal')
ax[2].set_ylabel('$45^\circ$', rotation = 'horizontal')
ax[3].set_ylabel('$67^\circ$', rotation = 'horizontal')
ax[4].set_ylabel('$90^\circ$', rotation = 'horizontal')
ax[4].set_xlabel('Time[ms]')
#  ax[4,0].set_xticks(timepoints, time)
#  ax[4,1].set_xticks(timepoints, time)
#  ax[4,2].set_xticks(timepoints, time)
ax[0].set_title('No ionic shift')
#  plt.tight_layout()
#  ax[4].legend(title = '$\Delta E_K$')
fig.savefig('Orientation_tuning', dpi = 200)
fig.savefig('FIG_4B_alt.svg', dpi = 400)
fig.savefig('FIG_4B_alt.pdf', dpi = 400)
plt.show()






