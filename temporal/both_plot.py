import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')


data1 = np.load('measure_NAK_new_19_300_0.npy')
data2 = np.load('measure_NAK_new_29_300_0.npy')
data3 = np.load('measure_NAK_new_39_300_0.npy')
data1in = np.load('measure_NAK_inn_19_300_0.npy')
data2in = np.load('measure_NAK_inn_29_300_0.npy')
data3in = np.load('measure_NAK_inn_39_300_0.npy')

data_null_1 = np.load('measure_NAK_new_19_300_none.npy')
data_null_2 = np.load('measure_NAK_new_29_300_none.npy')
data_null_3 = np.load('measure_NAK_new_39_300_none.npy')
data_null_1in = np.load('measure_NAK_inn_19_300_none.npy')
data_null_2in = np.load('measure_NAK_inn_29_300_none.npy')
data_null_3in = np.load('measure_NAK_inn_39_300_none.npy')

data_all_1 = np.load('measure_NAK_new_19_300_all.npy')
data_all_2 = np.load('measure_NAK_new_29_300_all.npy')
data_all_3 = np.load('measure_NAK_new_39_300_all.npy')
data_all_1in = np.load('measure_NAK_inn_19_300_all.npy')
data_all_2in = np.load('measure_NAK_inn_29_300_all.npy')
data_all_3in = np.load('measure_NAK_inn_39_300_all.npy')

#  data3 = np.load('measure_NAK_05_edge.npy')
data_arr = [data1, data2, data3]
data_null_arr = [data_null_1, data_null_2, data_null_3]

data_arr_in = [data1in, data2in, data3in]
data_null_arr_in = [data_null_1in, data_null_2in, data_null_3in]

data_all_arr = [data_all_1, data_all_2, data_all_3]
data_all_arr_in = [data_all_1in, data_all_2in, data_all_3in]

pmat = [1.9, 2.9, 3.9]

fig, ax = plt.subplots(3, 1, figsize = (17,6), sharex =  True, sharey = True)

for idx, data in enumerate(data_arr):

    data = data.reshape(10,4,-1)
    data_in = data_arr_in[idx]
    data_in = data_in.reshape(10,4,-1)
    data_in = np.mean(data_in, axis = 1)
    #  data_std = np.std(data, axis = 1)
    data = np.mean(data, axis = 1)
    time = np.arange(0,data.shape[1],1)*0.2

    DKE = data#/5
    DKE = -26.7*np.log((140)/(4 + 1*DKE))
    for i in range(6):
        ax[idx].plot(time,DKE[i,:] + 95, color = plt.cm.copper_r(abs(i-5)/5) )
        #  ax[idx].plot(time,DKE[i,:] + 95, color = plt.cm.copper(abs(i)/10) )
    ax[idx].set_title('$K_{dec} = $'+ str(pmat[idx]) + ' E-8 $m/s$') 
    ax[idx].set_ylabel('$\Delta E_{K^+} [mV]$', rotation = 90)


for idx, data in enumerate(data_null_arr):
    data = data.reshape(10,4,-1)
    data_std = np.std(data, axis = 1) 
    data = np.mean(data, axis = 1)
    data_in = data_null_arr_in[idx]
    data_in = data_in.reshape(10,4,-1)
    data_in = np.mean(data_in, axis = 1)
    time = np.arange(0,data.shape[1],1)*0.2
    DKE = data#/5
    #  DKE = -26.7*np.log(140/(4 + 1*DKE))
    DKE = -26.7*np.log((140)/(4 + 1*DKE))
    #  ax[idx].plot(time, DKE[0,:] + 95, color = 'green', ls = '--', label = 'Lower reference $\Delta E_K$')
    ax[idx].grid('x')


ax[0].legend()

#  ax[idx].set(xlabel = 'Time [ms]', ylabel = '$\Delta E_{K^+}$', title = '$\Delta E_{K^+}$ changes along the dendrite')
ax[-1].set_xlabel('Time[ms]')

sm = plt.cm.ScalarMappable(cmap='copper_r', norm=plt.Normalize(vmin=0, vmax=50))
sm.set_array([])  # Set an empty array

# Add a colorbar
#  cbar = plt.colorbar(sm, ax=ax[0])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Distance from cluster segment $[\mu m]$')


#  fig.suptitle('$\Delta E_{K^+}$ changes along the dendrite \n at different $K^+$ pump strengths')
fig.savefig('Temporal_plot_KPUMP', dpi = 400)

#  fig, ax = plt.subplots(2, 1, figsize = (15,5), sharex =  True, sharey = True)
fig, ax = plt.subplots(1, 1, figsize = (15,3), sharex =  True, sharey = True)
for idx, data in enumerate(data_arr[1:2]):
    data = data.reshape(10,4,-1)
    data_in = data_arr_in[idx]
    data_in = data_in.reshape(10,4,-1)
    data_in = np.mean(data_in, axis = 1)
    #  data_std = np.std(data, axis = 1)
    data = np.mean(data, axis = 1)
    time = np.arange(0,data.shape[1],1)*0.2

    DKE = data#/5
    DKE_only = -26.7*np.log((140)/(4 + 1*DKE))
    DKE = -26.7*np.log((140 - data_in)/(4 + 1*DKE))
    for i in range(6):
        if i == 5:
            ax.plot(time,DKE[i,:] + 95, color = plt.cm.copper_r(abs(i-5)/5), label = 'Intra- and extra-cellular' )
            ax.plot(time,DKE_only[i,:] + 95, color = plt.cm.copper_r(abs(i-5)/5), ls = '--' ,label = 'Extracellular only')
        else:
            ax.plot(time,DKE[i,:] + 95, color = plt.cm.copper_r(abs(i-5)/5) )
            ax.plot(time,DKE_only[i,:] + 95, color = plt.cm.copper_r(abs(i-5)/5), ls = '--' )

        #  ax[1].plot(time,DKE_only[i,:] + 95, color = plt.cm.copper_r(abs(i-5)/5) )
        #  ax[idx].plot(time,DKE[i,:] + 95, color = plt.cm.copper(abs(i)/10) )
    #  ax.set_title('$K_{dec} = $'+ str(pmat[1]) + ' E-8 $m/s$')
    ax.legend()
    ax.set_ylabel('$\Delta E_{K^+} [mV]$', rotation = 90)
    ax.grid('x')
    ax.set_xlabel('Time [ms]')
plt.tight_layout()

#  for idx, data in enumerate(data_all_arr[1:2]):
#      data = data.reshape(10,4,-1)
#      data_std = np.std(data, axis = 1)
#      data = np.mean(data, axis = 1)
#      data_in = data_all_arr_in[idx]
#      data_in = data_in.reshape(10,4,-1)
#      data_in = np.mean(data_in, axis = 1)
#      time = np.arange(0,data.shape[1],1)*0.2
#      DKE = data#/5
#      #  DKE = -26.7*np.log(140/(4 + 1*DKE))
#      DKE = -26.7*np.log((140)/(4 + 1*DKE))
#      ax.plot(time, DKE[3,:] + 95, color = 'red', ls = '--', label = 'Upper reference $\Delta E_K$')
#      ax.grid('x')
#
#  for idx, data in enumerate(data_null_arr[1:2]):
#      data = data.reshape(10,4,-1)
#      data_std = np.std(data, axis = 1)
#      data = np.mean(data, axis = 1)
#      data_in = data_null_arr_in[idx]
#      data_in = data_in.reshape(10,4,-1)
#      data_in = np.mean(data_in, axis = 1)
#      time = np.arange(0,data.shape[1],1)*0.2
#      DKE = data#/5
#      #  DKE = -26.7*np.log(140/(4 + 1*DKE))
#      DKE = -26.7*np.log((140)/(4 + 1*DKE))
#      ax.plot(time, DKE[0,:] + 95, color = 'green', ls = '--', label = 'Lower reference $\Delta E_K$')
#      ax.set_ylabel('$\Delta E_{K^+} [mV]$', rotation = 45)
#      ax.legend()

#  fig.suptitle('$\Delta E_{K^+}$ changes along the dendrite \n with different simulation regimes')
fig.savefig('Temporal_plot_regime', dpi = 400)

plt.show()
