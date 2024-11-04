import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')


def load_data(kind, angle, k_max):
    V_arr = []
    del_arr = []
    w_arr = []
    for i in range(10):
        data = np.load(f'../EK_effect/long_clamp_data_{i}_{kind}_{angle}_{k_max}.npy', allow_pickle = True)
        #  data = np.load(f'./EK_effect/EK_effect/no_noise_clamp_data_{i}_{kind}_{angle}_{k_max}.npy', allow_pickle = True)
        #  data = np.load(f'./EK_short/no_noise_clamp_data_{i}_{kind}_{angle}_{k_max}.npy', allow_pickle = True)
        data = np.load(f'./single_traces/data_{i}_{kind}_{angle}_{k_max}.npy')
        V = data.item().get('V')
        RT = data.item().get('RT')
        V_arr.append(V)
        del_arr.append(data.item().get('delays'))
        w_arr.append(data.item().get('EK'))
    V_arr = np.vstack(V_arr)
    return V_arr,del_arr,w_arr, RT,

#  C_00_4, DC_00_4, wDC_00_4,  RT = load_data('clu', 0, 4)
#  C_00_6, DC_00_6, wDC_00_6,  RT = load_data('clu', 0, 6)
#  C_00_8, DC_00_8, wDC_00_8,  RT = load_data('clu', 0, 8)
#  C_00_10, DC_00_10, wDC_00_10,  RT = load_data('clu', 0, 10)

C_00_6, DC_00_6, wDC_00_6,  RT = load_data('clu', 0, 6)
C_00_8, DC_00_8, wDC_00_8,  RT = load_data('clu', 0, 8)
C_00_10, DC_00_10, wDC_00_10,  RT = load_data('clu', 0, 10)
C_00_12, DC_00_12, wDC_00_12,  RT = load_data('clu', 0, 12)
C_00_16, DC_00_16, wDC_00_16,  RT = load_data('clu', 0, 16)
C_00_18, DC_00_18, wDC_00_18,  RT = load_data('clu', 0, 18)

#  C_00_10, DC_00_10, wDC_00_10,  RT = load_data('clu', 0, 10)
#  C_30_4, DC_00_4, wDC_00_4,  RT = load_data('clu', 30, 4)
#  C_30_6, DC_00_6, wDC_00_6,  RT = load_data('clu', 30, 6)
#  C_30_8, DC_00_8, wDC_00_8,  RT = load_data('clu', 30, 8)
#  C_30_10, DC_00_10, wDC_00_10,  RT = load_data('clu', 30, 10)
#
#  C_15_4, DC_00_4, wDC_00_4,  RT = load_data('clu', 15, 4)
#  C_15_6, DC_00_6, wDC_00_6,  RT = load_data('clu', 15, 6)
#  C_15_8, DC_00_8, wDC_00_8,  RT = load_data('clu', 15, 8)
#  C_15_10, DC_00_10, wDC_00_10,  RT = load_data('clu', 15, 10)


def find_tau(data, RT, threshold = -30):
    t_dur = np.zeros((10,2)) + 1e-6
    for i in range(data.shape[0]):
        trace = data[i,:]
        trace_0 = trace[:50000]
        trace_1 = trace[50000:100000]
        #  trace_2 = trace[100000:150000]
        if np.max(trace_0) > threshold: # Traces bellow thr is just ignored, as it it not considered a spike
            first = (np.where(trace_0 > threshold)[0][0])
            last = np.where(trace_0[first + 100:] < threshold)[0][0]
            t_dur[i,0] = RT[last + first + 100] - RT[first]

        if np.max(trace_1) > threshold: # Traces bellow thr is just ignored, as it it not considered a spike
            first = (np.where(trace_1 > threshold)[0][0])
            last = np.where(trace_1[first + 100:] < threshold)[0][0]
            t_dur[i,1] = RT[last + first + 100] - RT[first]

    print(np.round(t_dur,2))

    t_diff = t_dur.copy()
    #  t_diff = t_diff[:,1:]
    #  t_diff[:,0] -= t_dur[:,0]
    
    #  t_diff[:,0] /= t_dur[:,0]
    #  t_diff[t_diff > 1] = 1

    t_diff_mean = t_diff.mean(axis = 1)
    t_diff_err = t_diff.std(axis = 1)/np.sqrt(2)


    #  return t_dur, t_diff, t_diff_mean, t_diff_err
    return t_diff[:,0], t_diff[:,1], t_diff_mean, t_diff_err

k_max = np.array([1,2,4,6,8])


#  t_C00_4, delta_C00_4 , tmean_C00_4 , terr_C00_4 = find_tau(C_00_4, RT)
#  t_C00_6, delta_C00_6 , tmean_C00_6 , terr_C00_6 = find_tau(C_00_6, RT)
#  t_C00_8, delta_C00_8 , tmean_C00_8 , terr_C00_8 = find_tau(C_00_8, RT)
#  t_C00_10, delta_C00_10 , tmean_C00_10 , terr_C00_10 = find_tau(C_00_10, RT)
t_C00_6, delta_C00_6 , tmean_C00_6 , terr_C00_6 = find_tau(C_00_6, RT)
t_C00_8, delta_C00_8 , tmean_C00_8 , terr_C00_8 = find_tau(C_00_8, RT)
t_C00_10, delta_C00_10 , tmean_C00_10 , terr_C00_10 = find_tau(C_00_10, RT)
t_C00_12, delta_C00_12 , tmean_C00_12 , terr_C00_12 = find_tau(C_00_12, RT)
t_C00_16, delta_C00_16 , tmean_C00_16 , terr_C00_16 = find_tau(C_00_16, RT)
t_C00_18, delta_C00_18 , tmean_C00_18 , terr_C00_18 = find_tau(C_00_18, RT)

#  t_C15_4, delta_C15_4 , tmean_C15_4 , terr_C15_4 = find_tau(C_15_4, RT)
#  t_C15_6, delta_C15_6 , tmean_C15_6 , terr_C15_6 = find_tau(C_15_6, RT)
#  t_C15_8, delta_C15_8 , tmean_C15_8 , terr_C15_8 = find_tau(C_15_8, RT)
#  t_C15_10, delta_C15_10 , tmean_C15_10 , terr_C15_10 = find_tau(C_15_10, RT)
#
#  t_C30_4, delta_C30_4 , tmean_C30_4 , terr_C30_4 = find_tau(C_30_4, RT)
#  t_C30_6, delta_C30_6 , tmean_C30_6 , terr_C30_6 = find_tau(C_30_6, RT)
#  t_C30_8, delta_C30_8 , tmean_C30_8 , terr_C30_8 = find_tau(C_30_8, RT)
#  t_C30_10, delta_C30_10 , tmean_C30_10 , terr_C30_10 = find_tau(C_30_10, RT)
#
fig, ax = plt.subplots(1,2, figsize = (10,6))



#  fig, ax = plt.subplots(1,1, figsize = (10,7), sharey = True)
idx0 = []
idx4 = []
idx10= []
time = np.linspace(0,500,50000)
for i in range(10):
    #  ax[0].plot((C_00_10[i,:50000]),  alpha = .2, color = 'hotpink')
    if np.max(C_00_18[i,:50000]) > -8:
        idx = np.argmax(C_00_18[i, :50000])
        val = np.roll(C_00_18[i,:50000], 25000 - idx)
        idx0.append(val)

        #  ax[0].plot((C_00_18[i,:50000]),  alpha = .2, color = 'hotpink')
        #  ax[0].plot(time, val,  alpha = .2, color = 'hotpink')
    if np.max(C_00_18[i,50000:100000]) > -8:
        idx = np.argmax(C_00_18[i, 50000:100000])
        val = np.roll(C_00_18[i,50000:100000], 25000 - idx)
        idx10.append(val)
        #  ax[0].plot((C_00_10[i,50000:100000]),  alpha = .2, color = 'teal')
        #  ax[0].plot(time,val,  alpha = .2, color = 'teal')
    if np.max(C_00_6[i,50000:100000]) > -8:
        idx = np.argmax(C_00_6[i, 50000:100000])
        val = np.roll(C_00_6[i,50000:100000], 25000 - idx)
        idx4.append(val)
        #  ax[0].plot(time,val,  alpha = .2, color = 'goldenrod')
        #  ax[0].plot((C_00_4[i,50000:100000]),  alpha = .2, color = 'goldenrod')
    #  ax[0].plot((C_00_10[i,50000:100000]),  alpha = .2, color = 'teal')
    #  ax[0].plot((C_00_4[i,50000:100000]),  alpha = .2, color = 'goldenrod')

idx0 = np.vstack(idx0)
idx10 = np.vstack(idx10)
idx4 = np.vstack(idx4)

ax[0].plot(time,np.mean(idx0, axis =0),  alpha = 1, color = 'hotpink', label = '$0 mV$')
ax[0].plot(time,np.mean(idx4, axis = 0),  alpha = 1, color = 'goldenrod', label = '$6 mV$')
ax[0].plot(time,np.mean(idx10, axis = 0),  alpha = 1, color = 'teal', label = '$18 mV$')
#  ax[0].plot(np.mean(C_00_10[idx0,:50000], axis =0),  alpha = 1, color = 'hotpink', label = '$0 mV$')
#  ax[0].plot(np.mean(C_00_4[idx4,50000:100000], axis = 0),  alpha = 1, color = 'goldenrod', label = '$4 mV$')
#  ax[0].plot(np.mean(C_00_10[idx10,50000:100000], axis = 0),  alpha = 1, color = 'teal', label = '$10 mV$')
ax[0].set_xlim(200,430)
ax[0].hlines(-30, 200, 430, ls = '--', color = 'grey')
ax[0].legend(title = 'Mean trace')
ax[0].text(310, -30, 'Vm threshold', color = 'grey', size = 20)
ax[0].set(xlabel = 'Time [ms]', ylabel = 'Vm [mV]', title = 'Example traces at $\Delta\Theta = 0^\circ$')

ax[0].text(-0.1, 1.05, 'C', fontweight = 'bold', transform = ax[0].transAxes, fontsize = 20)
ax[1].text(-0.1, 1.05, 'D', fontweight = 'bold', transform = ax[1].transAxes, fontsize = 20)
for i in range(10):
    if t_C00_6[i] < 10:
        t_C00_6[i] = np.nan
        delta_C00_6[i] = np.nan
    if t_C00_8[i] <10:
        t_C00_8[i] = np.nan
        delta_C00_8[i] = np.nan
    if t_C00_10[i] <10:
        t_C00_10[i] = np.nan
        delta_C00_10[i] = np.nan
    if t_C00_12[i] <10:
        t_C00_12[i] = np.nan
        delta_C00_12[i] = np.nan
    if t_C00_16[i] <10:
        t_C00_16[i] = np.nan
        delta_C00_16[i] = np.nan
    if t_C00_18[i] <10:
        t_C00_18[i] = np.nan
        delta_C00_18[i] = np.nan


mean_before_00 = np.array([np.nanmean(t_C00_6),
                           np.nanmean(t_C00_8),
                           np.nanmean(t_C00_10),
                           np.nanmean(t_C00_12),
                           np.nanmean(t_C00_16),
                           np.nanmean(t_C00_18)])

mean_after_00 = np.array([np.nanmean(mean_before_00),
                          np.nanmean(delta_C00_6),
                          np.nanmean(delta_C00_8),
                          np.nanmean(delta_C00_10),
                          np.nanmean(delta_C00_12),
                          np.nanmean(delta_C00_16),
                          np.nanmean(delta_C00_18)]) 

std_before_00 = np.array([np.nanstd(t_C00_6),
                          np.nanstd(t_C00_8),
                          np.nanstd(t_C00_10),
                          np.nanstd(t_C00_12),
                          np.nanstd(t_C00_16),
                          np.nanstd(t_C00_18)])/np.sqrt(10)

std_after_00 = np.array([np.nanstd(t_C00_10),
                         np.nanstd(delta_C00_6),
                         np.nanstd(delta_C00_8),
                         np.nanstd(delta_C00_10),
                         np.nanstd(delta_C00_12),
                         np.nanstd(delta_C00_16),
                         np.nanstd(delta_C00_18)])/np.sqrt(10)

#  DEK = np.array([0, 4,6,8,10])
DEK = np.array([0,6,8,10, 12, 16, 18])
ax[1].errorbar(DEK, mean_after_00, yerr = std_after_00, color = 'black', label = '$\Delta E_K shifted$', ls = '--', marker = 'o', capsize = 4)
#  ax[1].legend(title = 'Run')
ax[1].set(xlabel = 'Change in EK [mV]', ylabel = ' Plateau duration over -30mV [ms]', title = 'Change in NMDA plateau length at $\Delta\Theta = 0^\circ$', ylim = (30, 140))

plt.savefig('NMDA_plateau', dpi=200)
plt.savefig('FIG_4CD.svg', dpi=400)
plt.savefig('FIG_4CD.pdf', dpi=400)

plt.show()

#  fig, ax = plt.subplots(1,1, figsize = (4,6))
#  ax.plot(RT[:20000], C_00_10[3,70000:90000], color = 'black', lw = 4)
#  ax.errorbar(15, -25, yerr = 27, capsize = 10, capthick = 3, ls = '--', lw = 3, color = 'red', label = 'Spike Height')
#  ax.errorbar(75, -30, xerr = 45, capsize = 10, capthick = 3, ls = '--', lw = 3, color = 'green', label = 'Spike Plateau')
#  ax.legend()
#  ax.set_xticks([])
#  ax.set_yticks([])
#  ax.set_ylabel('Vm', size = 15)
#  ax.set_xlabel('Time', size = 15)
#  fig.savefig('Illustrative')

plt.show()


