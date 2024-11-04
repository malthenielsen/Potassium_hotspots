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
        V_arr.append(np.load(f'data_{i}_{kind}_{angle}_{sur}.npy'))


    V_arr = np.vstack(V_arr)
    RT = np.arange(len(V_arr[0]))

    return V_arr, RT


C_ba, RT = load_data('clu',0, 'GABA_FALSE')
C_hw, RT = load_data('clu',5, 'GABA_FALSE')
C_df, RT = load_data('clu',10, 'GABA_FALSE')
C_wf, RT = load_data('clu',15, 'GABA_FALSE')
C_twf, RT = load_data('clu',20, 'GABA_FALSE')
#  C_twf, RT = load_data('clu',25, 'GABA_FALSE')
#  C_twf, RT = load_data('clu',10, 'quad_weight')
             

fig = plt.figure(figsize = (8, 10), layout = 'tight')
gs  = gridspec.GridSpec(5, 4)
ax00 = fig.add_subplot(gs[:1, :3])
ax01 = fig.add_subplot(gs[:1, 3:])
ax10 = fig.add_subplot(gs[1:2, :3], sharey = ax00)
ax11 = fig.add_subplot(gs[1:2, 3:], sharey = ax01)
ax20 = fig.add_subplot(gs[2:3, :3], sharey = ax00)
ax21 = fig.add_subplot(gs[2:3, 3:], sharey = ax01)
ax30 = fig.add_subplot(gs[3:4, :3], sharey = ax00)
ax31 = fig.add_subplot(gs[3:4, 3:], sharey = ax01)
ax40 = fig.add_subplot(gs[4:5, :3], sharey = ax00)
ax41 = fig.add_subplot(gs[4:5, 3:], sharey = ax01)
ax01.spines['left'].set_color('plum')
ax01.spines['right'].set_color('plum')
ax01.spines['left'].set_linewidth(2)
ax01.spines['right'].set_linewidth(2)
ax01.spines['left'].set_linestyle('--')
ax01.spines['right'].set_linestyle('--')
ax11.spines['left'].set_color('plum')
ax11.spines['right'].set_color('plum')
ax11.spines['left'].set_linewidth(2)
ax11.spines['right'].set_linewidth(2)
ax11.spines['left'].set_linestyle('--')
ax11.spines['right'].set_linestyle('--')
ax21.spines['left'].set_color('plum')
ax21.spines['right'].set_color('plum')
ax21.spines['left'].set_linewidth(2)
ax21.spines['right'].set_linewidth(2)
ax21.spines['left'].set_linestyle('--')
ax21.spines['right'].set_linestyle('--')
ax31.spines['left'].set_color('plum')
ax31.spines['right'].set_color('plum')
ax31.spines['left'].set_linewidth(2)
ax31.spines['right'].set_linewidth(2)
ax31.spines['left'].set_linestyle('--')
ax31.spines['right'].set_linestyle('--')
ax41.spines['left'].set_color('plum')
ax41.spines['right'].set_color('plum')
ax41.spines['left'].set_linewidth(2)
ax41.spines['right'].set_linewidth(2)
ax41.spines['left'].set_linestyle('--')
ax41.spines['right'].set_linestyle('--')
#  ax00.text(-0.1, 1.15, 'B', fontweight = 'bold', transform = ax00.transAxes, fontsize = 20)
#  ax00.vlines(1200, -70, 0, ls = '--', lw = 2, color = 'plum')
#  ax00.vlines(1400, -70, 0, ls = '--', lw = 2, color = 'plum')
#  ax10.vlines(1200, -70, 0, ls = '--', lw = 2, color = 'plum')
#  ax10.vlines(1400, -70, 0, ls = '--', lw = 2, color = 'plum')
#  ax20.vlines(1200, -70, 0, ls = '--', lw = 2, color = 'plum')
#  ax20.vlines(1400, -70, 0, ls = '--', lw = 2, color = 'plum')
#  ax30.vlines(1200, -70, 0, ls = '--', lw = 2, color = 'plum')
#  ax30.vlines(1400, -70, 0, ls = '--', lw = 2, color = 'plum')
#  ax40.vlines(1200, -70, 0, ls = '--', lw = 2, color = 'plum')
#  ax40.vlines(1400, -70, 0, ls = '--', lw = 2, color = 'plum')
ax00.fill_between([1200, 1400],[-70, -70], [0,0], color = 'tab:red', alpha = .2)
ax10.fill_between([1200, 1400],[-70, -70], [0,0], color = 'tab:red', alpha = .2)
ax20.fill_between([1200, 1400],[-70, -70], [0,0], color = 'tab:red', alpha = .2)
ax30.fill_between([1200, 1400],[-70, -70], [0,0], color = 'tab:red', alpha = .2)
ax40.fill_between([1200, 1400],[-70, -70], [0,0], color = 'tab:red', alpha = .2)

ax00.fill_between([200, 400],[-70, -70], [0,0], color = 'tab:blue', alpha = .2)
ax10.fill_between([200, 400],[-70, -70], [0,0], color = 'tab:blue', alpha = .2)
ax20.fill_between([200, 400],[-70, -70], [0,0], color = 'tab:blue', alpha = .2)
ax30.fill_between([200, 400],[-70, -70], [0,0], color = 'tab:blue', alpha = .2)
ax40.fill_between([200, 400],[-70, -70], [0,0], color = 'tab:blue', alpha = .2)

ax00.text(50,-15, '$0^\circ$' , size = 15)
ax10.text(50,-15, '$5^\circ$', size = 15)
ax20.text(50,-15, '$10^\circ$', size = 15)
ax30.text(50,-15, '$15^\circ$', size = 15)
ax40.text(50,-15, '$20^\circ$', size = 15)

for i in range(10):
    ax00.plot(RT[:], C_ba[i,:], color = 'grey', alpha = .3, linewidth = .5)
    ax10.plot(RT[:], C_hw[i,:], color = 'grey', alpha = .3, linewidth = .5)
    ax20.plot(RT[:], C_df[i,:], color = 'grey', alpha = .3, linewidth = .5)
    ax30.plot(RT[:], C_wf[i,:], color = 'grey', alpha = .3, linewidth = .5)
    ax40.plot(RT[:], C_twf[i,:], color = 'grey', alpha = .3, linewidth = .5)


    #  ax01.plot(RT[1200:1400], C_00[i,1200:1400], color = 'grey', alpha = .3, linewidth = .5)
    #  ax11.plot(RT[1200:1400], C_22[i,1200:1400], color = 'grey', alpha = .3, linewidth = .5)
    #  ax21.plot(RT[1200:1400], C_45[i,1200:1400], color = 'grey', alpha = .3, linewidth = .5)
    #  ax31.plot(RT[1200:1400], C_67[i,1200:1400], color = 'grey', alpha = .3, linewidth = .5)
    #  ax41.plot(RT[1200:1400], C_90[i,1200:1400], color = 'grey', alpha = .3, linewidth = .5)

ax00.plot(RT[:], np.mean(C_ba, axis = 0), color = 'teal', alpha = 1, linewidth = 1)
ax10.plot(RT[:], np.mean(C_hw, axis = 0), color = 'teal', alpha = 1, linewidth = 1)
ax20.plot(RT[:], np.mean(C_df, axis = 0), color = 'teal', alpha = 1, linewidth = 1, label = 'Mean trace \n N = 10')
ax30.plot(RT[:], np.mean(C_wf, axis = 0), color = 'teal', alpha = 1, linewidth = 1)
ax40.plot(RT[:], np.mean(C_twf, axis = 0), color = 'teal', alpha = 1, linewidth = 1)

ax01.plot(RT[1200:1400], np.mean(C_ba[:,1200:1400], axis = 0), color = 'tab:red', alpha = 1, linewidth = 1)
ax11.plot(RT[1200:1400], np.mean(C_hw[:,1200:1400], axis = 0), color = 'tab:red', alpha = 1, linewidth = 1)
ax21.plot(RT[1200:1400], np.mean(C_df[:,1200:1400], axis = 0), color = 'tab:red', alpha = 1, linewidth = 1)
ax31.plot(RT[1200:1400], np.mean(C_wf[:,1200:1400], axis = 0), color = 'tab:red', alpha = 1, linewidth = 1)
ax41.plot(RT[1200:1400], np.mean(C_twf[:,1200:1400], axis = 0), color = 'tab:red', alpha = 1, linewidth = 1)

ax01.plot(RT[1200:1400], np.mean(C_ba[:,200:400], axis = 0), color = 'tab:blue', alpha = 1, linewidth = 1)
ax11.plot(RT[1200:1400], np.mean(C_hw[:,200:400], axis = 0), color = 'tab:blue', alpha = 1, linewidth = 1)
ax21.plot(RT[1200:1400], np.mean(C_df[:,200:400], axis = 0), color = 'tab:blue', alpha = 1, linewidth = 1)
ax31.plot(RT[1200:1400], np.mean(C_wf[:,200:400], axis = 0), color = 'tab:blue', alpha = 1, linewidth = 1)
ax41.plot(RT[1200:1400], np.mean(C_twf[:,200:400], axis = 0), color = 'tab:blue', alpha = 1, linewidth = 1)

#  ax00.set_title('Stim orientation $0^\circ$')
#  ax10.set_title('Stim orientation $5^\circ$')
#  ax20.set_title('Stim orientation $10^\circ$')
#  ax30.set_title('Stim orientation $15^\circ$')
#  ax40.set_title('Stim orientation $20^\circ$')

ax01.set(ylim =(-70,0), xlabel = 'Time (ms)')
ax11.set(ylim =(-70,0), xlabel = 'Time (ms)')
ax21.set(ylim =(-70,0), xlabel = 'Time (ms)')
ax31.set(ylim =(-70,0), xlabel = 'Time (ms)')
ax41.set(ylim =(-70,0), xlabel = 'Time (ms)')
ax00.set(ylim =(-70,0), xlabel = 'Time (ms)', ylabel = 'Vm (mV)')
ax10.set(ylim =(-70,0), xlabel = 'Time (ms)', ylabel = 'Vm (mV)')
ax20.set(ylim =(-70,0), xlabel = 'Time (ms)', ylabel = 'Vm (mV)')
ax30.set(ylim =(-70,0), xlabel = 'Time (ms)', ylabel = 'Vm (mV)')
ax40.set(ylim =(-70,0), xlabel = 'Time (ms)', ylabel = 'Vm (mV)')
ax20.legend(loc = 6)

fig.suptitle('Clustered type point dendrite Vm traces \n at different stimulation angle relative to soma prefered')
fig.suptitle('Point dendrite Vm traces at different stimulation orientations relative to target orientation')
plt.show()
#  fig.savefig('Traces?plot', dpi = 200)
#  fig.savefig('FIG_2B.svg', dpi = 400)
#  fig.savefig('FIG_S2B_GABA_random.png', dpi = 200)
fig.savefig('NO_GABA.png', dpi = 200)

exit()

