import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def load_data(kind, angle, sur):
    V_arr = []
    del_arr = []
    w_arr = []
    for i in range(10):
        V_arr.append(np.load(f'traces_2B/85data_{i}_{kind}_{angle}_{sur}.npy'))


    V_arr = np.vstack(V_arr)
    RT = np.arange(len(V_arr[0]))

    return V_arr, RT


C_ba, RT = load_data('clu',0, 'freq')
C_hw, RT = load_data('clu',22.5, 'freq')
C_df, RT = load_data('clu',45, 'freq')
C_wf, RT = load_data('clu',67.5, 'freq')
C_twf, RT = load_data('clu',90, 'freq')
             

fig = plt.figure(figsize = (8, 10), layout = 'tight')
gs  = gridspec.GridSpec(5, 4)
ax00 = fig.add_subplot(gs[:1,  :])
ax10 = fig.add_subplot(gs[1:2, :], sharey = ax00)
ax20 = fig.add_subplot(gs[2:3, :], sharey = ax00)
ax30 = fig.add_subplot(gs[3:4, :], sharey = ax00)
ax40 = fig.add_subplot(gs[4:5, :], sharey = ax00)
ax00.axis('off')
ax10.axis('off')
ax20.axis('off')
ax30.axis('off')
#  ax40.axis('off')

ax00.fill_between([770, 800],[-70, -70], [0,0], color = 'tab:orange', alpha = .2)
ax10.fill_between([770, 800],[-70, -70], [0,0], color = 'tab:orange', alpha = .2)
ax20.fill_between([770, 800],[-70, -70], [0,0], color = 'tab:orange', alpha = .2)
ax30.fill_between([770, 800],[-70, -70], [0,0], color = 'tab:orange', alpha = .2)
ax40.fill_between([770, 800],[-70, -70], [0,0], color = 'tab:orange', alpha = .2)

ax00.fill_between([270, 300],[-70, -70], [0,0], color = 'tab:orange', alpha = .2)
ax10.fill_between([270, 300],[-70, -70], [0,0], color = 'tab:orange', alpha = .2)
ax20.fill_between([270, 300],[-70, -70], [0,0], color = 'tab:orange', alpha = .2)
ax30.fill_between([270, 300],[-70, -70], [0,0], color = 'tab:orange', alpha = .2)
ax40.fill_between([270, 300],[-70, -70], [0,0], color = 'tab:orange', alpha = .2)

ax00.fill_between([1270, 1300],[-70, -70], [0,0], color = 'tab:orange', alpha = .2)
ax10.fill_between([1270, 1300],[-70, -70], [0,0], color = 'tab:orange', alpha = .2)
ax20.fill_between([1270, 1300],[-70, -70], [0,0], color = 'tab:orange', alpha = .2)
ax30.fill_between([1270, 1300],[-70, -70], [0,0], color = 'tab:orange', alpha = .2)
ax40.fill_between([1270, 1300],[-70, -70], [0,0], color = 'tab:orange', alpha = .2)


for i in range(10):
    ax00.plot(RT[:], C_ba[i,:], color = 'grey', alpha = .3, linewidth = .7)
    ax10.plot(RT[:], C_hw[i,:], color = 'grey', alpha = .3, linewidth = .7)
    ax20.plot(RT[:], C_df[i,:], color = 'grey', alpha = .3, linewidth = .7)
    ax30.plot(RT[:], C_wf[i,:], color = 'grey', alpha = .3, linewidth = .7)
    ax40.plot(RT[:], C_twf[i,:],color = 'grey', alpha = .3, linewidth = .7)


ax00.plot(RT[:], np.mean(C_ba, axis = 0), color = 'black', alpha = 1, linewidth = 1)
ax10.plot(RT[:], np.mean(C_hw, axis = 0), color = 'black', alpha = 1, linewidth = 1)
ax20.plot(RT[:], np.mean(C_df, axis = 0), color = 'black', alpha = 1, linewidth = 1, label = 'Mean trace \n N = 10')
ax30.plot(RT[:], np.mean(C_wf, axis = 0), color = 'black', alpha = 1, linewidth = 1)
ax40.plot(RT[:], np.mean(C_twf, axis = 0), color = 'black', alpha = 1, linewidth = 1)


ax00.set_title('$0^\circ$')
ax10.set_title('$22.5^\circ$')
ax20.set_title('$45^\circ$')
ax30.set_title('$67.5^\circ$')
ax40.set_title('$90^\circ$')
#  ax00.set_title('Baseline')
#  ax10.set_title('Double weight')
#  ax20.set_title('Double frequency')
#  ax30.set_title('Double weight and frequency')
#  ax40.set_title('Triple weight and frequency')

ax00.set(ylim =(-70,0), xlabel = 'Time (ms)', ylabel = 'Vm (mV)')
ax10.set(ylim =(-70,0), xlabel = 'Time (ms)', ylabel = 'Vm (mV)')
ax20.set(ylim =(-70,0), xlabel = 'Time (ms)', ylabel = 'Vm (mV)')
ax30.set(ylim =(-70,0), xlabel = 'Time (ms)', ylabel = 'Vm (mV)')
ax40.set(ylim =(-70,0), xlabel = 'Time (ms)', ylabel = 'Vm (mV)')
#  ax20.legend(loc = 6)

fig.suptitle('Clustered type point dendrite Vm traces \n at different stimulation angle relative to soma prefered')
fig.suptitle('Point dendrite Vm traces at different stimulation orientations relative to target orientation')
plt.show()
#  fig.savefig('Traces?plot', dpi = 200)
#  fig.savefig('FIG_2B.svg', dpi = 400)
#  fig.savefig('FIG_S2B_GABA_random.png', dpi = 200)
fig.savefig('fig2B.pdf', dpi = 400)

exit()

