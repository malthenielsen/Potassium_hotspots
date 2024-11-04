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
    fnames = glob(path)
    fnames.sort(key=os.path.getctime)

    fnames_trunk = fnames
    #  fnames_soma = fnames[1::2]
    #  fnames_trunk = fnames[::2]



    data = np.zeros((3,22,len(fnames_trunk)))


    for l, fname in enumerate(fnames_trunk):
        trunk = np.load(fname)
        bin_ = []
        for i in range(len(trunk)):
            inds, _ = find_peaks(trunk[i], height = -30)
            print(len(trunk[i]))
            #  for idx in inds:
                #  trunk[i][idx - 55:idx + 55] = np.mean(trunk[i][idx+55:idx+60])
            if i%3 == 0:
                baseline = np.mean(trunk[i][2500:3000])
            trunk_tmp = trunk[i][8000:36000] - baseline
            trunk_tmp[trunk_tmp > 30] = 30
            area = np.trapz(trunk_tmp, dx = .001)
            bin_.append(area)

        data[:,:,l] = np.array(bin_).reshape(22,3).T

    t_test = np.zeros((2,22))
    residuals_arr = np.zeros((2,22,len(fnames_trunk)))
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

    print(T_large)

    mean = np.mean(data, axis = 2)
    eom = np.std(data, axis = 2)/np.sqrt(data.shape[2])
    print(mean)
    return mean, eom, T_large, R_val, residuals_arr

#  paths = ['./tuning4_frac/*.npy']
#  paths = ['./3frac_test/*.npy']
paths = ['./folder_auc/*.npy']

def get_traces(path):
    fnames = glob(path)
    fnames.sort(key=os.path.getctime)

    fnames_soma = fnames[1::2]
    fnames_trunk = fnames[::2]
    inds = [9]
    return_list = []
    for l, fname in enumerate(fnames_soma):
        if l in inds:
            return_list.append(np.load(fname))
    return return_list



#  paths = ['../longer_trunk/many_runs/*.npy']
for path in paths:
    means, eoms, T_val, R_val, residuals = func(path)

def OSI(arr, ori):
    ori = np.deg2rad(ori)
    top = np.sum(arr * np.exp(2*1j*ori))
    F = top/np.sum(arr)
    return 1 - np.arctan2(np.imag(F), np.real(F))

#  angles =np.arange(0,90,4)
#  angles = np.array([0,5,10,15,20,30,40,50,80,89])
angles = np.linspace(0,35,18)
angles = np.append(angles, np.array([40,50,75,90]))

print(OSI(means[0,:], angles))
print(OSI(means[1,:], angles))
print(OSI(means[2,:], angles))
#  exit()

def chi_square(constant, y, y_err):
    return np.sum((y - constant)**2/y_err)

def lfunc(x,a, b):
    return a*x + b

def altfunc(x, a):
    return a/x + 1


#  angles = np.arange(0,90,4).astype(float)

fig_residuals, ax_res = plt.subplots(1,1, figsize = (8,6))
#  residuals
ax_res.hlines(0, 0, 90, ls = 'dashed', color = 'black', alpha = .7)
ax_res.errorbar(angles, np.mean(residuals[0,:,:], axis = 1),np.std(residuals[0,:,:], axis = 1)/np.sqrt(residuals.shape[2]), color = 'hotpink', ls = '', marker = 'o', capsize = 4, label  = '5mV')
ax_res.errorbar(angles, np.mean(residuals[1,:,:], axis = 1),np.std(residuals[1,:,:], axis = 1)/np.sqrt(residuals.shape[2]), color = 'teal', ls = '', marker = 'o', capsize = 4, label = '10mV')
ax_res.legend(title = '$\Delta E_K$')
#  ax_res.set_ylim(-1, 2)
ax_res.set(ylabel = '$\Delta$ firing', xlabel = r'$\Delta\theta_{soma prefered}$', title = 'Residual change, caused by k_ext')
ax_res.text(40, -.7, f'Paired one tailed Students T test \n P-value \n 5mV = {T_val[0]} \n 10mV = {T_val[1]}', size = 13)
fig_residuals.savefig('residuals_aoc', dpi = 200)

errors = np.zeros((2,22))
mean = np.zeros((2,22))

#  np.save('neuron_eom', eoms)
#  np.save('neuron_mean', means)

mean_10_data = np.zeros_like(mean)
mean_0_data = np.zeros_like(mean)
error_10_data = np.zeros_like(mean)
error_0_data = np.zeros_like(mean)

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
        z = r_mean_10 / r_mean_0
        errors[j,i] = np.std(z)
        #  errors[j,i] = np.sqrt(eom_10[i]**2 + eom_0[i]**2)
        mean[j,i] = np.mean(z)
        #  mean_10_data[i] = mean_10[i]
        #  mean_0_data[i] = mean_0[i]
        #  error_10_data[i] = eom_10[i]
        #  error_0_data[i] = eom_0[i]


from scipy.optimize import minimize

fig, ax = plt.subplots(1,1, figsize = (8,6))
ax.set_ylim(.5, 2)

ax.errorbar(angles, mean[0,:], yerr = errors[0,:], ls = 'dashed', marker = 'o', capsize = 4, label = r'$\frac{Soma_{5} FR}{Soma_{0} FR}$' )
ax.errorbar(angles, mean[1,:], yerr = errors[1,:], ls = 'dashed', marker = 'o', capsize = 4, label = r'$\frac{Soma_{10} FR}{Soma_{0} FR}$' )
ax.set(xlabel ='Angle from soma pref', ylabel = 'Fraction', title = '$\Delta E_{K}$ impact style')



#  res = minimize(chi_square, .1, args = (mean, errors))
#  print(stats.chi2.sf(chi_square(res.x[0],mean, errors), mean.shape[0]-1))
#  chi_1 = stats.chi2.sf(chi_square(res.x[0],mean, errors), mean.shape[0]-1)
#  finished = False
#  sig = 0
#  chi2 = chi_square(res.x[0],mean, errors)
#  while not finished:
#      delta = chi_square(res.x[0]+sig,mean, errors) - chi2
#      if delta > 1:
#          finished = True
#          print(res.x, 'pm', sig)
#      sig += .001
#
for i in range(2):
    chi2_fit = Chi2Regression(altfunc, means[0,:10], mean[i,:10], errors[i,:10])
    minuit_chi2 = Minuit(chi2_fit, a = .1)
    minuit_chi2.errordef = 1
    minuit_chi2.migrad()
    #  ax.plot(angles, lfunc(angles, *minuit_chi2.values[:]), label = 'Fitted line')
    print(minuit_chi2.values[:], 'eps')
    print(minuit_chi2.errors[:])
    print(stats.chi2.sf(minuit_chi2.fval, 10 - 1))


    chi2_fit = Chi2Regression(lfunc, angles[:10], mean[i,:10], errors[i,:10])
    minuit_chi2 = Minuit(chi2_fit, a = .1, b = 1)
    minuit_chi2.errordef = 1
    minuit_chi2.migrad()
    ax.plot(angles, lfunc(angles, *minuit_chi2.values[:]), label = 'Fitted line')
    print(minuit_chi2.values[:])
    print(minuit_chi2.errors[:])
    print(stats.chi2.sf(minuit_chi2.fval, 10 - 2))
d = {
        'Slope = ': [minuit_chi2.values['a'],minuit_chi2.errors['a']],
        'Offset = ': [minuit_chi2.values['b'],minuit_chi2.errors['b']],
    }
text = nice_string_output(d, extra_spacing=2, decimals=3)
add_text_to_ax(0.1, 0.7, text, ax, fontsize=14)
#  chi_2 = stats.chi2.sf(minuit_chi2.fval, mean.shape[0]-2)
#  print(chi_1, chi_2)
#  print(res.x[0], minuit_chi2.fval)
#  ax.text(40, -.4, f'$\chi^2$ = {chi_1}', size = 13)

ax.legend()
#  plt.savefig('Multiplicity', dpi = 200)

#  fig_residuals, ax_res = plt.subplots(1,1, figsize = (8,6))
#  fig_tuning, ax_tune = plt.subplots(1,1, figsize = (8,6))
#  fig_setup, ax_setup = plt.subplots(1,1, figsize = (8,6))


#  residuals = resid[0]
#  ax_res.hlines(0, 0, 90, ls = 'dashed', color = 'black', alpha = .7)
#  ax_res.errorbar(angles, np.mean(residuals[1,:,:], axis = 1),np.std(residuals[1,:,:], axis = 1)/np.sqrt(residuals.shape[2]), color = 'teal', ls = '', marker = 'o', capsize = 4)
#  ax_res.set_ylim(-1, 1)
#  ax_res.set(ylabel = '$\Delta$ firing', xlabel = r'$\Delta\theta_{soma prefered}$', title = 'Residual change, caused by k_ext')
#  ax_res.text(40, -.4, f'Paired one tailed Students T test \n P-value \t   {t_big[1][1]} \n T-value \t {R_val[1][0]} \n Ddof \t {R_val[1][1]}', size = 13)
#
#  ax_setup.errorbar(angles, mean_10_data[2,:], error_10_data[2,:], ls = 'dashed', marker = 'o', capsize = 4, label = 'Global K')
#  ax_setup.errorbar(angles, mean_10_data[0,:], error_10_data[0,:], ls = 'dashed', marker = 'o', capsize = 4, label = 'Local K')
#  ax_setup.errorbar(angles, mean_0_data[0,:], error_10_data[0,:], ls = 'dashed', marker = 'o', capsize = 4, label = 'No K')
#  ax_setup.legend(title = 'Neuron setup')
#  ax_setup.set(ylabel = 'Firing rate', xlabel = r'$\Delta\theta_{soma prefered}$', title = 'Soma firingrate')
#kjj


import matplotlib.gridspec as gridspec
fig = plt.figure(tight_layout = True, figsize = (7,6))
gs = gridspec.GridSpec(3,4)
ax_tune = fig.add_subplot(gs[:,:])
#  ax_A = fig.add_subplot(gs[0,3])
#  ax_A.set_xticks([])
#  ax_A.set_yticks([])
#  ax_A.set_ylim(-70,60)
#  ax_B = fig.add_subplot(gs[1,3])
#  ax_B.set_xticks([])
#  ax_B.set_yticks([])
#  ax_B.set_ylim(-70,60)
#  ax_C = fig.add_subplot(gs[2,3])
#  ax_C.set_xticks([])
#  ax_C.set_yticks([])
#  ax_C.set_ylim(-70,60)
#  ax_A.text(3000,20, 'A', fontsize = 25)
#  ax_B.text(3000,20, 'B', fontsize = 25)
#  ax_C.text(3000,20, 'C', fontsize = 25)
#
#
#  idx = [0, 1, 2]
traces = get_traces(path)
traces = traces[0].reshape(22,3,-1)
#  ax_A.plot(traces[ 0,  idx,25000:31000].T)
#  ax_B.plot(traces[ 5, idx,25000:31000].T)
#  ax_C.plot(traces[ 15, idx,25000:31000].T)

#  ax_tune.text(angles[0], means[0,0] - 1, 'A', fontsize = 25)
#  ax_tune.text(angles[5] - 5, means[0,0] - 1, 'B', fontsize = 25)
#  ax_tune.text(angles[15], means[0,15] + .5, 'C', fontsize = 25)


ax_tune.errorbar(angles,  means[0,:], eoms[0,:], ls = 'solid', marker = '', capsize = 0, label = 'No shift', color = 'darkgrey')
ax_tune.errorbar(angles,  means[1,:], eoms[1,:], ls = 'solid', marker = '', capsize = 0, label = 'Low shift', color = 'dodgerblue')
ax_tune.errorbar(angles,  means[2,:], eoms[2,:], ls = 'solid', marker = '', capsize = 0, label = 'High shift', color = 'goldenrod')
ax_tune.legend(title = '$\Delta E_K$')
ax_tune.set(ylabel = 'Bulk signal strength', xlabel = r'$\Delta\theta_{soma prefered}$', title = 'Vm AOC')
ax_tune.text(-0.1, 1.05, 'F', fontweight = 'bold', transform = ax_tune.transAxes, fontsize = 20)
plt.savefig('FIG_4F.svg', dpi = 400)
plt.savefig('FIG_4F.pdf', dpi = 400)



fig = plt.figure(tight_layout = True, figsize = (7,6))
gs = gridspec.GridSpec(3,15)
ax_tune = fig.add_subplot(gs[:,:10])
ax_tune_late = fig.add_subplot(gs[:,11:], sharey = ax_tune)
ax_tune_late.spines['left'].set_visible(False)
#  ax_tune_late.set_yticks([])

ax_tune.errorbar(angles[:18],  means[0,:18], eoms[0,:18], ls = 'solid', marker = '', capsize = 0, label = 'No shift', color = 'darkgrey')
ax_tune.errorbar(angles[:18],  means[1,:18], eoms[1,:18], ls = 'solid', marker = '', capsize = 0, label = 'Low shift', color = 'dodgerblue')
ax_tune.errorbar(angles[:18],  means[2,:18], eoms[2,:18], ls = 'solid', marker = '', capsize = 0, label = 'High shift', color = 'goldenrod')
ax_tune.set_xlim(0,35)

ax_tune_late.errorbar(angles[20:],  means[0,20:], eoms[0,20:], ls = 'solid', marker = '', capsize = 0, label = 'No shift', color = 'darkgrey')
ax_tune_late.errorbar(angles[20:],  means[1,20:], eoms[1,20:], ls = 'solid', marker = '', capsize = 0, label = 'Low shift', color = 'dodgerblue')
ax_tune_late.errorbar(angles[20:],  means[2,20:], eoms[2,20:], ls = 'solid', marker = '', capsize = 0, label = 'High shift', color = 'goldenrod')
ax_tune_late.set_xlim(80,90)
ax_tune_late.set_xticks([80,85,90], [80,85,90])

ax_tune.legend(title = '$\Delta E_K$')
ax_tune.set(ylabel = 'Firing rate', xlabel = r'$\Delta\theta_{soma prefered}$', title = 'Soma firingrate')
ax_tune.text(-0.1, 1.05, 'C', fontweight = 'bold', transform = ax_tune.transAxes, fontsize = 20)
plt.savefig('FIG_4F_split.svg', dpi = 400)
plt.savefig('FIG_4F_split.pdf', dpi = 400)
plt.show()









#  plt.savefig('FIG_4F.svg', dpi = 400)

#  print(np.arange(0,90,4))
#  print(np.arange(0,23,1))
#  time = np.arange(00,550,50)
#  timepoints = np.arange(0000,55000,5000)
#  fig, ax = plt.subplots(7,1, figsize = (6,10), sharex = True, sharey = True)
#  ax[0].plot(traces[0, 2, 20000:70000], color = 'red',  lw=1.5)
#  ax[1].plot(traces[4, 2, 20000:70000], color = 'red',  lw=1.5)
#  ax[2].plot(traces[8, 2, 20000:70000], color = 'red',  lw=1.5)
#  ax[3].plot(traces[11, 2, 20000:70000], color = 'red', lw=1.5)
#  ax[4].plot(traces[15, 2, 20000:70000], color = 'red', lw=1.5)
#  ax[5].plot(traces[19, 2, 20000:70000], color = 'red', lw=1.5)
#  ax[6].plot(traces[22, 2, 20000:70000], color = 'red', lw=1.5, label = '10 mV')
#  ax[0].plot(traces[0, 0, 20000:70000], color = 'black',  lw=1.5)
#  ax[1].plot(traces[4, 0, 20000:70000], color = 'black',  lw=1.5)
#  ax[2].plot(traces[8, 0, 20000:70000], color = 'black',  lw=1.5)
#  ax[3].plot(traces[11, 0, 20000:70000], color = 'black', lw=1.5)
#  ax[4].plot(traces[15, 0, 20000:70000], color = 'black', lw=1.5)
#  ax[5].plot(traces[19, 0, 20000:70000], color = 'black', lw=1.5)
#  ax[6].plot(traces[22, 0, 20000:70000], color = 'black', lw=1.5, label = '0 mV')
#  ax[0].set_ylabel('$0^\circ$', rotation = 'horizontal')
#  ax[1].set_ylabel('$15^\circ$', rotation = 'horizontal')
#  ax[2].set_ylabel('$30^\circ$', rotation = 'horizontal')
#  ax[3].set_ylabel('$45^\circ$', rotation = 'horizontal')
#  ax[4].set_ylabel('$60^\circ$', rotation = 'horizontal')
#  ax[5].set_ylabel('$75^\circ$', rotation = 'horizontal')
#  ax[6].set_ylabel('$90^\circ$', rotation = 'horizontal')
#  ax[6].set_xlabel('Time[ms]')
#  ax[6].set_xticks(timepoints, time)
#  ax[6].legend(title = '$\Delta E_K$')
#  fig.savefig('Orientation_tuning', dpi = 300)


plt.show()







