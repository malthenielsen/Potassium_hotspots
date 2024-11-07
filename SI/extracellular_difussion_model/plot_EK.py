import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')



fire_1 = np.load('data/fire_0.0.npy')
fire_2 = np.load('data/fire_22.5.npy')
fire_3 = np.load('data/fire_45.0.npy')
fire_w = np.load('data/fire_weight.npy')

#  data1 = np.load('measure_NAK_onoff_29_300_0.npy')
#  data1 = np.load('data_cell/freqmeasure_NAK_onoff_29_300_0.npy')
#  data2 = np.load('data_cell/freqmeasure_NAK_onoff_29_300_22.npy')
#  data3 = np.load('data_cell/freqmeasure_NAK_onoff_29_300_45.npy')
data1 = np.load('data/freqmeasure_NAK_onoff_29_300_0.npy')
data2 = np.load('data/freqmeasure_NAK_onoff_29_300_22.npy')
data3 = np.load('data/freqmeasure_NAK_onoff_29_300_45.npy')
data11 = np.load('data/measure_NAK_onoff_29_300_0_1.npy')
data21 = np.load('data/measure_NAK_onoff_29_300_22_1.npy')
data31 = np.load('data/measure_NAK_onoff_29_300_45_1.npy')
data_arr = [data1, data2, data3]
data_arr_alt = [data11, data21, data31]
fig, ax = plt.subplots(3, 1, figsize = (15,7), sharex =  True, sharey = True)
pmat = [0, 22.5, 45]
for idx, data in enumerate(data_arr):
    data = data.reshape(10,4,-1)
    data = np.mean(data, axis = 1)
    time = np.arange(0,data.shape[1],1)*0.2
    DKE = data#/5

    DKE = -26.7*np.log(140/(4 + 1*DKE))
    for i in range(6):
        ax[idx].plot(time,DKE[i,:] + 95, color = plt.cm.copper_r(abs(i-5)/5), ls = '--' )

    data = data_arr_alt[idx].reshape(10,4,-1)
    data = np.mean(data, axis = 1)
    time = np.arange(0,data.shape[1],1)*0.2
    DKE = data#/5

    DKE = -26.7*np.log(140/(4 + 1*DKE))
    for i in range(6):
        ax[idx].plot(time,DKE[i,:] + 95, color = plt.cm.copper_r(abs(i-5)/5) )
        #  ax[idx].plot(time,DKE[i,:] + 95, color = plt.cm.copper(abs(i)/10) )
    #  ax[idx].set_title('Stimulus interval = '+ str(pmat[idx]) + '  $(ms)$')
    ax[idx].set_title('Stimulation orientation = '+ str(pmat[idx]) + '$^\circ$')

    ax[idx].set_ylabel('$\Delta E_{K^+} (mV)$', rotation = 90)
    ax[idx].grid('x')

ax[0].set_xlim(0,2000)
ax[1].set_ylim(-3,22)
ax[2].set_xlabel('Time (ms)')
for i in range(3):
    ax[i].spines['bottom'].set_position('zero')
for f in fire_1:
    ax[0].vlines(f, -3, -1.5, color = 'grey', lw = 0.3)
for f in fire_2:
    ax[1].vlines(f, -3, -1.5, color = 'grey', lw = 0.3)
for f in fire_3:
    ax[2].vlines(f, -3, -1.5, color = 'grey', lw = 0.3)
for f in fire_w:
    for i in range(3):
        ax[i].vlines(f, 0, -1.5, color = 'black', lw = 0.1)


sm = plt.cm.ScalarMappable(cmap='copper_r', norm=plt.Normalize(vmin=0, vmax=50))
sm.set_array([])  # Set an empty array

# Add a colorbar
#  cbar = plt.colorbar(sm, ax=ax[0])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Distance from cluster segment $(\mu m)$')

#  fig.suptitle('$\Delta E_K$ shift at different interstimulus intervals')
fig.savefig('S15.png', dpi = 300)
plt.show()
#  exit()

data1 = np.load('data/measure_NAK_night_29_200_0_0.npy')
data2 = np.load('data/measure_NAK_night_29_300_0_0.npy')
data3 = np.load('data/measure_NAK_night_29_400_0_0.npy')
#  data1 = np.load('measure_NAK_new_29_200_0.npy')
#  data2 = np.load('measure_NAK_new_29_300_0.npy')
#  data3 = np.load('measure_NAK_new_29_400_0.npy')
data_arr = [data1, data2, data3]
fig, ax = plt.subplots(3, 1, figsize = (15,7), sharex =  True, sharey = True)
pmat = [200, 300, 400]
for idx, data in enumerate(data_arr):
    data = data.reshape(10,4,-1)
    data_std = np.std(data, axis = 1) 
    data = np.mean(data, axis = 1)
    time = np.arange(0,data.shape[1],1)*0.2
    #  for i in range(10):
        #  plt.plot(time,data[i,:], color = plt.cm.inferno(abs(i-5)/5))
        #  plt.fill_between(time, data[i,:], data[i,:] - data_std[i,:], data[i,:]+data_std[i,:],color = plt.cm.inferno(abs(i-5)/5))
    #  plt.show()
    #  print(data.shape)

#  DKE = data/np.max(data)
    DKE = data#/5

    DKE = -26.7*np.log(140/(4 + 1*DKE))
    for i in range(6):
        ax[idx].plot(time,DKE[i,:] + 95, color = plt.cm.copper_r(abs(i-5)/5) )
        #  ax[idx].plot(time,DKE[i,:] + 95, color = plt.cm.copper(abs(i)/10) )
    ax[idx].set_title('Stimulus interval = '+ str(pmat[idx]) + '  $(ms)$') 

    ax[idx].set_ylabel('$\Delta E_{K^+} (mV)$', rotation = 90)
    ax[idx].grid('x')

ax[0].set_xlim(0,1000)
ax[2].set_xlabel('Time (ms)')
ax[1].set_ylim(-3, 22)

sm = plt.cm.ScalarMappable(cmap='copper_r', norm=plt.Normalize(vmin=0, vmax=50))
sm.set_array([])  # Set an empty array

# Add a colorbar
#  cbar = plt.colorbar(sm, ax=ax[0])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Distance from cluster segment $(\mu m)$')

#  fig.suptitle('$\Delta E_K$ shift at different interstimulus intervals')
fig.savefig('Temporal_FR_test_29.png', dpi = 200)
plt.show()

#  exit()

data1 = np.load('data/measure_NAK_night_29_300_0_0.npy')
data2 = np.load('data/measure_NAK_night_29_300_22_0.npy')
data3 = np.load('data/measure_NAK_night_29_300_45_0.npy')
#  data1 = np.load('MIAmeasure_NAK_onoff_29_300_0_0.npy')
#  data2 = np.load('MIAmeasure_NAK_onoff_29_300_22_0.npy')
#  data3 = np.load('MIAmeasure_NAK_onoff_29_300_22_0.npy')
#  data1 = np.load('measure_NAK_new_29_300_0.npy')
#  data2 = np.load('30measure_NAK_new_29_300_22.npy')
#  data3 = np.load('measure_NAK_new_29_300_45.npy')
data_off = np.load('data/measure_NAK_night_29_300_none.npy')
#  data_off = np.load('30measure_NAK_new_29_300_none.npy')
data_arr = [data1, data2, data3]
#  fig, ax = plt.subplots(3, 1, figsize = (15,7), sharex =  True, sharey = True)

from matplotlib.gridspec import GridSpec
fig = plt.figure( figsize = (14,17),constrained_layout = True)
gs = GridSpec(30, 1, figure=fig, )
ax = []
ax.append(fig.add_subplot(gs[0:5, :]))
ax.append(fig.add_subplot(gs[5:10,:]))
ax.append(fig.add_subplot(gs[10:15,:]))
ax.append(fig.add_subplot(gs[15:20, :]))
ax.append(fig.add_subplot(gs[20:25,:]))
ax.append(fig.add_subplot(gs[25:,:]))
ax = np.array(ax)

#  ax_r = fig.add_subplot(gs[-1,:])
fire = np.load('data/fire_weight.npy')
for f in fire:
    #  ax_r.vlines(f, 0, 1, color = 'black', lw = 0.1)
    for i in range(3):
        ax[i].vlines(f, -1.5, 0, color = 'black', lw = 0.4)
    for i in range(3,6):
        ax[i].vlines(f, -0.5, 0, color = 'black', lw = 0.4)

for i in range(6):
    ax[i].spines['bottom'].set_position('zero')



pmat = [0, 22.5, 45]
for idx, data in enumerate(data_arr):
    data = data.reshape(10,4,-1)
    data_std = np.std(data, axis = 1) 
    data = np.mean(data, axis = 1)
    time = np.arange(0,data.shape[1],1)*0.2
    #  for i in range(10):
        #  plt.plot(time,data[i,:], color = plt.cm.inferno(abs(i-5)/5))
        #  plt.fill_between(time, data[i,:], data[i,:] - data_std[i,:], data[i,:]+data_std[i,:],color = plt.cm.inferno(abs(i-5)/5))
    #  plt.show()
    #  print(data.shape)

#  DKE = data/np.max(data)
    DKE = data#/5

    DKE = -26.7*np.log(140/(4 + 1*DKE))
    for i in range(6):
        ax[idx].plot(time,DKE[i,:] + 95, color = plt.cm.copper_r(abs(i-5)/5) )
        #  ax[idx].plot(time,DKE[i,:] + 95, color = plt.cm.copper(abs(i)/10) )
    ax[idx].set_title('Stimulation orientation = '+ str(pmat[idx]) + '$^\circ$')

    ax[idx].set_ylabel('$\Delta E_{K^+} (mV)$', rotation = 90)
    ax[idx].grid('xy')

ax[0].set_xlim(0,1000)
ax[1].set_xlim(0,1000)
ax[2].set_xlim(0,1000)
ax[0].set_ylim(-1.5,22.5)
ax[1].set_ylim(-1.5,22.5)
ax[2].set_ylim(-1.5,22.5)
ax[0].set_xticks([0, 200, 400, 600, 800, 1000],['', '', '', '','',''])
ax[1].set_xticks([0, 200, 400, 600, 800, 1000],['', '', '', '','',''])
ax[2].set_xticks([0, 200, 400, 600, 800, 1000],['', '', '', '','',''])
ax[3].set_xticks([0, 200, 400, 600, 800, 1000],['', '', '', '','',''])
#  ax_r.set_xlim(0,1000)

data_off = data_off.reshape(10,4,-1)
data_off_std = np.std(data_off, axis = 1)
data_off = np.mean(data_off, axis = 1)
time = np.arange(0,data_off.shape[1],1)*0.2
DKE = data_off#/5
DKE = -26.7*np.log(140/(4 + 1*DKE))
ax[0].plot(time, DKE[0,:] + 95, color = 'green', ls = '--', label = 'Reference $\Delta E_K$')
ax[1].plot(time, DKE[0,:] + 95, color = 'green', ls = '--', label = 'Reference $\Delta E_K$')
ax[2].plot(time, DKE[0,:] + 95, color = 'green', ls = '--', label = 'Reference $\Delta E_K$')
#  ax_r.set_xlabel('Time (ms)')
#  ax_r.set_yticks([])
#  ax_r.set_xticks([])

#  sm = plt.cm.ScalarMappable(cmap='copper_r', norm=plt.Normalize(vmin=0, vmax=50))
#  sm.set_array([])  # Set an empty array
#
#  # Add a colorbar
#  #  cbar = plt.colorbar(sm, ax=ax[0])
#  cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
#  cbar.set_label('Distance from cluster segment $(\mu m)$')
#
#  #  fig.suptitle('$\Delta E_K$ shift at different stimulus orientation')
#  fig.savefig('Temporal_SO_test_29.png', dpi = 200)
#  plt.show()

#  fig = plt.figure( figsize = (15,7))
#  gs = GridSpec(16, 1, figure=fig)
#  ax = []
#  ax.append(fig.add_subplot(gs[0:5, :]))
#  ax.append(fig.add_subplot(gs[5:10,:]))
#  ax.append(fig.add_subplot(gs[10:15,:]))
#  ax = np.array(ax)

#  ax_r = fig.add_subplot(gs[-1,:])
#  fire = np.load('fire.npy')
#  for f in fire:
    #  ax_r.vlines(f, 0, 1, color = 'black', lw = 0.1)



pmat = [0, 22.5, 45]
for idx, data in enumerate(data_arr):
    data = data.reshape(10,4,-1)
    data_std = np.std(data, axis = 1) 
    data = np.mean(data, axis = 1)
    time = np.arange(0,data.shape[1],1)*0.2
    #  for i in range(10):
        #  plt.plot(time,data[i,:], color = plt.cm.inferno(abs(i-5)/5))
        #  plt.fill_between(time, data[i,:], data[i,:] - data_std[i,:], data[i,:]+data_std[i,:],color = plt.cm.inferno(abs(i-5)/5))
    #  plt.show()
    #  print(data.shape)

#  DKE = data/np.max(data)
    DKE = data#/5

    #  DKE = -26.7*np.log(140/(4 + 1*DKE))
    for i in range(6):
        ax[idx+3].plot(time,DKE[i,:], color = plt.cm.copper_r(abs(i-5)/5) )
        #  ax[idx].plot(time,DKE[i,:] + 95, color = plt.cm.copper(abs(i)/10) )
    ax[idx+3].set_title('Stimulation orientation = '+ str(pmat[idx]) + '$^\circ$')

    ax[idx+3].set_ylabel('$\Delta K^+_O (mM)$', rotation = 90)
    ax[idx+3].grid('xy')

#  ax[5].grid('x')

ax[3].set_xlim(0,1000)
ax[4].set_xlim(0,1000)
ax[5].set_xlim(0,1000)
ax[3].set_ylim(-0.5,7)
ax[4].set_ylim(-0.5,7)
ax[5].set_ylim(-0.5,7)
ax[3].set_xticks([0, 200, 400, 600, 800, 1000],['', '', '', '','',''])
ax[4].set_xticks([0, 200, 400, 600, 800, 1000],['', '', '', '','',''])
#  ax[5].set_xticks([])
ax[3].set_facecolor('whitesmoke')
ax[4].set_facecolor('whitesmoke')
ax[5].set_facecolor('whitesmoke')
#  ax_r.set_xlim(0,1000)

data_off = data_off.reshape(10,4,-1)
data_off_std = np.std(data_off, axis = 1)
data_off = np.mean(data_off, axis = 1)
time = np.arange(0,data_off.shape[1],1)*0.2
DKE = data_off#/5
#  DKE = -26.7*np.log(140/(4 + 1*DKE))
#  ax[0].plot(time, DKE[0,:], color = 'green', ls = '--', label = 'Reference $\Delta E_K$')
#  ax[1].plot(time, DKE[0,:], color = 'green', ls = '--', label = 'Reference $\Delta E_K$')
#  ax[2].plot(time, DKE[0,:], color = 'green', ls = '--', label = 'Reference $\Delta E_K$')
#  ax_r.set_xlabel('Time (ms)')
#  ax_r.set_yticks([])
ax[5].set_xlabel('Time (ms)')
ax[0].text(-50, 20, 'a', fontsize = 20) 
ax[3].text(-50, 6, 'b', fontsize = 20) 

sm = plt.cm.ScalarMappable(cmap='copper_r', norm=plt.Normalize(vmin=0, vmax=50))
sm.set_array([])  # Set an empty array

# Add a colorbar
#  cbar = plt.colorbar(sm, ax=ax[0])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
cbar.set_label('Distance from cluster segment $(\mu m)$')

#  fig.suptitle('$\Delta E_K$ shift at different stimulus orientation')
fig.savefig('Temporal_SO_test_KO_below.png', dpi = 200)
plt.show()
#  data1 = np.load('measure_NAK_new_19.npy')
#  data2 = np.load('measure_NAK_new_29.npy')
#  data3 = np.load('measure_NAK_new_39.npy')
#  data_null_1 = np.load('measure_NAK_new_19_none.npy')
#  data_null_2 = np.load('measure_NAK_new_29_none.npy')
#  data_null_3 = np.load('measure_NAK_new_39_none.npy')
#  #  data3 = np.load('measure_NAK_05_edge.npy')
#  data_arr = [data1, data2, data3]
#  data_null_arr = [data_null_1, data_null_2, data_null_3]
#  pmat = [1.9, 2.9, 3.9]
#  #  data = np.load('measure_NAK_none.npy')
#  #  data = np.load('measure_NAK_none_away.npy')
#  #  data = np.load('measure_NAK.npy')
#
#  #  arr = np.arange(40)
#  #  arr = arr.reshape(10,4)
#  #  print(arr)
#
#  #  for i in range(40):
#  #      plt.plot(data[i,:])
#  #  #  plt.imshow(data, aspect = 'auto')
#  #  plt.show()
#
#  fig, ax = plt.subplots(3, 1, figsize = (15,7), sharex =  True, sharey = True)
#  for idx, data in enumerate(data_arr):
#      data = data.reshape(10,4,-1)
#      data_std = np.std(data, axis = 1)
#      data = np.mean(data, axis = 1)
#      time = np.arange(0,data.shape[1],1)*0.2
#      #  for i in range(10):
#          #  plt.plot(time,data[i,:], color = plt.cm.inferno(abs(i-5)/5))
#          #  plt.fill_between(time, data[i,:], data[i,:] - data_std[i,:], data[i,:]+data_std[i,:],color = plt.cm.inferno(abs(i-5)/5))
#      #  plt.show()
#      #  print(data.shape)
#
#  #  DKE = data/np.max(data)
#      DKE = data#/5
#      DKE = -26.7*np.log(140/(4 + 1*DKE))
#      for i in range(6):
#          ax[idx].plot(time,DKE[i,:] + 95, color = plt.cm.copper_r(abs(i-5)/5) )
#          #  ax[idx].plot(time,DKE[i,:] + 95, color = plt.cm.copper(abs(i)/10) )
#      ax[idx].set_title('$K_{dec} = $'+ str(pmat[idx]) + ' E-8 $m/s$')
#      ax[idx].set_ylabel('$\Delta E_{K^+} [mV]$', rotation = 45)
#
#
#  for idx, data in enumerate(data_null_arr):
#      data = data.reshape(10,4,-1)
#      data_std = np.std(data, axis = 1)
#      data = np.mean(data, axis = 1)
#      time = np.arange(0,data.shape[1],1)*0.2
#      DKE = data#/5
#      DKE = -26.7*np.log(140/(4 + 1*DKE))
#      ax[idx].plot(time, DKE[0,:] + 95, color = 'green', ls = '--', label = 'Reference $\Delta E_K$')
#
#  ax[0].legend()
#
#  #  ax[idx].set(xlabel = 'Time [ms]', ylabel = '$\Delta E_{K^+}$', title = '$\Delta E_{K^+}$ changes along the dendrite')
#  ax[-1].set_xlabel('Time[ms]')
#
#  sm = plt.cm.ScalarMappable(cmap='copper_r', norm=plt.Normalize(vmin=0, vmax=50))
#  sm.set_array([])  # Set an empty array
#
#  # Add a colorbar
#  #  cbar = plt.colorbar(sm, ax=ax[0])
#  cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
#  cbar.set_label('Distance from cluster segment $[\mu m]$')
#
#
#  fig.suptitle('$\Delta E_{K^+}$ changes along the dendrite \n at different $K^+$ pump strengths')
#  fig.savefig('Temporal_plot_KPUMP', dpi = 400)


#  plt.tight_layout()
