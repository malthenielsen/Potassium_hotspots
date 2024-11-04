import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.gridspec import GridSpec

#  data_78 = np.load('IV_data_EK-78.npy')
#  data_80 = np.load('IV_data_EK-80.npy')
#  data_74 = np.load('IV_data_EK-74.npy')
#  data_70 = np.load('IV_data_EK-70.npy')
labels = ['Na_V', 'K_V','K_M', 'Ca_HVA', 'Ca_LVA', 'K_CaV', 'Ih', 'K_leak', 'Ca', 'KA_V'] 
labels = ['Na_V',
   'K_V',
   'KM_V',
   'Ca_HVA',
   'Ca_LVA',
   'K_Ca',
   'Ih',
   'K_Leak',
   'Ca',
   'KA_V',]

def IV(data, idx):
    V = data[idx][:,0]
    inward = [1, 4, 5]
    outward = [2,3,6,8,10]
    inward  = np.sum(data[idx][:,inward], axis = 1 )*2*np.pi*1e-4
    outward = np.sum(data[idx][:,outward], axis = 1)*2*np.pi*1e-4
    mg_block = 1/(1 + 2/3.57 *np.exp(-0.15*V))
    syn_act = 2.9e-3*mg_block*(V - 0)
    ampa = .32e-4*(V-0)
    I_nmda = outward+inward+syn_act + 1e-6*V 

    ax2.plot(V,outward+inward+syn_act + 1e-6*V, label = 'Sum of Intrensic and Extrensic')
    #  ax2.hlines(0, min(V), max(V), color = 'black')
    ax2.set_xlim(-90, 0)
    ax2.set_ylim(-.005,.005)
    ax2.set_xlabel('Voltage [mV]')
    ax2.set_ylabel('I [pA]')
    for k in range(0, len(V)-1, 10):
        xt = V[k]
        y0 = I_nmda[k]
        y2 = I_nmda[k]
        dx = V[k+1] - V[k]
        dy0 = (I_nmda[k+1]) - (I_nmda[k])
        dy2 = (I_nmda[k+1]) - (I_nmda[k])
        factor = 1
        if y0 > 0:
            factor = -1
        
        ax2.arrow(xt, y0, dx*factor, dy0*factor,  color = 'black', width = 1e-6, shape ='full', head_width = 2e-4, head_length = .5)
        #  ax2.arrow(xt, y2, dx, dy2,  color = 'teal')
    #  ax.legend()
    #  plt.show()

    #  return V, outward + inward #, syn_act

def ampa_plot(data, idx, amp = 0, nmda = 0, nmda_m =1):
    V = data[idx][:,0]
    inward = [1, 4, 5]
    outward = [2,3,6,8,10]
    inward  = np.sum(data[idx][:,inward], axis = 1 )*2*np.pi*1e-4
    outward = np.sum(data[idx][:,outward], axis = 1)*2*np.pi*1e-4
    mg_block = 1/(1 + 2/3.57 *np.exp(-0.15*V))
    syn_act = 2.9e-3*mg_block*(V - 0)*nmda*nmda_m
    ampa = .21e-4*(V-0)*amp
    I_nmda = outward+inward+syn_act + 1e-6*V + ampa 
    return V, I_nmda

def threshold(data, idx):
    V = data[idx][:,0]
    inward = [1, 4, 5]
    outward = [2,3,6,8,10]
    inward  = np.sum(data[idx][:,inward], axis = 1 )*2*np.pi*1e-4
    outward = np.sum(data[idx][:,outward], axis = 1)*2*np.pi*1e-4
    mg_block = 1/(1 + 2/3.57 *np.exp(-0.15*V))
    syn_act = 2.9e-3*mg_block*(V - 0)
    ampa_gbar = np.linspace(.1e-4, .5e-4, 1000)
    thr_V = 0
    ampa_need = 0
    for gbar in ampa_gbar:
        netto_current = outward+inward+syn_act+gbar*(V-0) + 1e-6*V
        netto_current = netto_current[:-20]
        #  print(V[-20])
        #  print(np.max(netto_current), gbar)
        if not np.any(netto_current > 0):
            thr_V = V[np.argmax(netto_current)]
            ampa_need = gbar
            break
    netto_current = outward+inward+syn_act + 1e-6*V
    for i in range(len(V)):
        if netto_current[i] > 0:
            thr_V_base = V[i]
            break
    return thr_V, ampa_need, thr_V_base



#  fig, ax = plt.subplots(1,1, figsize = (10,7))
fnames = glob.glob('./*.npy')
data = []
eks = []
fnames = ['./IV_data_EK-80.0.npy','./IV_data_EK-64.npy']
for fname in fnames:
    data.append(np.load(fname))
    ek = float(fname.split('-')[-1][:-4])*(-1)
    eks.append(ek)



V_bin = []
ampa_bin = []
V_base = []
#  fig2, ax2 = plt.subplots(1,1, figsize = (8,6))
#  fig, ax = plt.subplots(2,2, figsize = (15,8))
#  for i in range(20):
#      #  IV(data, i)
#      thr_V, thr_ampa, thr_V_base = threshold(data, i)
#      V_bin.append(thr_V)
#      ampa_bin.append(thr_ampa)
#      V_base.append(thr_V_base)
#  V_bin = np.array(V_bin) + 80
eks = np.array(eks) + 80
#  V_base = np.array(V_bin) + 80

#  ax[0, 0].scatter(eks, np.array(V_base) - np.array(V_bin), color = 'darkorange', label = 'Vm distance between the two points')
#  ax[0, 0].set_title('Vm distance to threshold')
#  ax[0, 0].set_ylabel('$\Delta Vm$')
#  ax[0, 0].set_xlabel('$\Delta$EK [mV]')
#  ax[0,0].legend()

V, n_cur_80c = ampa_plot(data, 0, nmda = 0)
V, n_cur_70c = ampa_plot(data, -1, nmda = 0)

V, n_cur_80 = ampa_plot(data, 0, nmda = 1)
V, n_cur_70 = ampa_plot(data, -1, nmda = 1)
V, n_cur_70_a = ampa_plot(data, -1, 1, 1)
V, n_cur_70_n = ampa_plot(data, -1, 0, 1, .85)

V, n_cur_80_a = ampa_plot(data, 0, .5, 1)
V, n_cur_80_n = ampa_plot(data, 0, 0, 1, .85)


fig = plt.figure(constrained_layout = False, figsize = (20,10))
gs = GridSpec(2,14)
ax00 = fig.add_subplot(gs[0,:4])
ax01 = fig.add_subplot(gs[0,4:8])
ax10 = fig.add_subplot(gs[1,:4])
ax11 = fig.add_subplot(gs[1,4:8])

ax2 = fig.add_subplot(gs[:,-6:-3])
ax3 = fig.add_subplot(gs[:,-3:])

ax01.plot(V, n_cur_80c, color = 'hotpink', label = '$\Delta$EK = 0')
ax01.plot(V, n_cur_70c, color = 'teal', label = '$\Delta$EK = 16')
ax01.set_ylim(-0.003, 0.005)
#  ax01.hlines(0, -100, 20, color = 'black')
ax01.set_xlim(-85,0)
ax01.legend()
ax01.set_title('Intrinsic currents')
axins = zoomed_inset_axes(ax01, 1.2, loc = 8)
axins.plot(V[10:33], np.zeros(23), color = 'tab:olive', ls = '--', lw = 2)
axins.plot(V[10:33], n_cur_80c[10:33], color = 'hotpink')
axins.plot(V[10:33], n_cur_70c[10:33], color = 'teal')
axins.scatter(-78, 0, color = 'hotpink', marker = 'o', label = 'stable fixpoints')
axins.scatter(-63, 0, color = 'teal', marker = 'o', label = 'stable fixpoints')
ax01.scatter(-63, 0, color = 'teal', marker = 'o', label = 'stable fixpoints')
ax01.scatter(-78, 0, color = 'hotpink', marker = 'o', label = 'stable fixpoints')
axins.spines[['right', 'top']].set_visible(True)
axins.set_yticks([])
axins.set_xlabel('Vm [mV]')
mark_inset(ax01, axins, loc1 = 3, loc2 = 1)
ax01.set_yticks([])
ax11.set_yticks([])
ax01.set_xticks([])
ax00.set_xticks([])
ytick = np.linspace(-0.003, 0.005,9, endpoint = True)
ax00.set_yticks(ytick, -ytick)

ytick = np.linspace(-0.004, 0.002,7, endpoint = True)
ax10.set_yticks(ytick, -ytick)

ax11.hlines(0, -100, 20, color = 'tab:purple', lw = 3, ls = '--')
ax01.hlines(0, -100, 20, color = 'tab:olive', lw = 3, ls = '--')
ax10.hlines(0, -100, 20, color = '#E3735E', lw = 3, ls = '--')


for k in range(0, len(V)-1, 6):
    xt = V[k]
    y0 = n_cur_80c[k]
    dx = V[k+1] - V[k]
    dy0 = (n_cur_80c[k+1]) - (n_cur_80c[k])
    factor = 1
    if y0 > 0:
        factor = -1
    ax01.arrow(xt, y0, dx*factor, dy0*factor,  color = 'hotpink', width = 1e-6, shape ='full', head_width = 2e-4, head_length = 1)

for k in range(0, len(V)-1, 6):
    xt = V[k]
    y0 = n_cur_70c[k]
    dx = V[k+1] - V[k]
    dy0 = (n_cur_70c[k+1]) - (n_cur_70c[k])
    factor = 1
    if y0 > 0:
        factor = -1
    ax01.arrow(xt, y0, dx*factor, dy0*factor,  color = 'teal', width = 1e-6, shape ='full', head_width = 2e-4, head_length = 1)

ax10.plot(V, n_cur_80, color = 'hotpink', label = '$\Delta$EK = 0')
ax10.plot(V, n_cur_70, color = 'teal', label = '$\Delta$EK = 16')
ax10.set_ylim(-0.004, 0.002)
#  ax10.hlines(0, -100, 20, color = 'black')
ax10.set_xlim(-80,0)
ax10.legend()
ax10.set_xlabel('Membrane Vm')
ax10.set_ylabel('I[A]')
ax10.set_title('Intrinsic currents and NMDA')
for k in range(0, len(V)-1, 6):
    xt = V[k]
    y0 = n_cur_80[k]
    dx = V[k+1] - V[k]
    dy0 = (n_cur_80[k+1]) - (n_cur_80[k])
    factor = 1
    if y0 > 0:
        factor = -1
    ax10.arrow(xt, y0, dx*factor, dy0*factor,  color = 'hotpink', width = 1e-6, shape ='full', head_width = 2e-4, head_length = 1)

for k in range(0, len(V)-1, 6):
    xt = V[k]
    y0 = n_cur_70[k]
    dx = V[k+1] - V[k]
    dy0 = (n_cur_70[k+1]) - (n_cur_70[k])
    factor = 1
    if y0 > 0:
        factor = -1
    ax10.arrow(xt, y0, dx*factor, dy0*factor,  color = 'teal', width = 1e-6, shape ='full', head_width = 2e-4, head_length = 1)

#  ax[1,0].plot(V, n_cur_80_a, color = 'hotpink', label = '$\Delta$EK = 0')
#  ax[1,0].plot(V, n_cur_70_a, color = 'teal', label = '$\Delta$EK = 10')

ax00.plot(V, n_cur_80c, color = 'tab:green', label = 'None')
ax00lab = [ '0', r'$\frac{1}{5}$', r'$\frac{2}{5}$', r'$\frac{3}{5}$',r'$\frac{4}{5}$', '1']
for i in range(6):
    V, n_cur_80_a = ampa_plot(data, 0, 0.1 + .3*i, 1)
    #  if i == 0:
        #  ax[1,0].plot(V, n_cur_80_a, plt.cm.cool(i/5), label = 'No AMPA')#, alpha = 0.1 + 0.15*i)
    #  elif i == 5:
        #  ax[1,0].plot(V, n_cur_80_a, plt.cm.cool(i/5), label = 'Full AMPA')#, alpha = 0.1 + 0.15*i)
    #  else:
    if i == 0:
        color = 'tab:blue'
    else:
        color = plt.cm.Reds(i/5)
    ax00.plot(V, n_cur_80_a, color = color, label = ax00lab[i])#, alpha = 0.1 + 0.15*i)

    for k in range(0, len(V)-1, 6):
        xt = V[k]
        y0 = n_cur_80_a[k]
        dx = V[k+1] - V[k]
        dy0 = (n_cur_80_a[k+1]) - (n_cur_80_a[k])
        factor = 1
        if y0 > 0:
            factor = -1
        ax00.arrow(xt, y0, dx*factor, dy0*factor,  color = color, width = 1e-6, shape ='full', head_width = 2e-4, head_length = 1)


#  hdl = ax00.scatter(np.zeros(5), np.zeros(5), color = plt.cm.Reds(np.linspace(0,1,5, endpoint = True)))
cbaxes = inset_axes(ax00, width="40%", height="5%", loc=2)
plt.colorbar(plt.cm.ScalarMappable(norm = None, cmap = 'Reds'), cax=cbaxes, ticks=[0,1], orientation='horizontal', label = 'AMPA strength')


ax00.set_ylim(-0.003, 0.005)
ax00.hlines(0, -100, 20, color = 'black')
ax00.set_xlim(-80,0)
ax00.legend(['Intrinsic ion channels', 'Intrinsic ion channels + NMDA'],ncol =1, loc = 10, bbox_to_anchor = (0.2, 0.7))
ax00.set_title('Current-Voltage landscapes')
ax00.set_ylabel('I [A]')

for k in range(0, len(V)-1, 6):
    xt = V[k]
    y0 = n_cur_80c[k]
    dx = V[k+1] - V[k]
    dy0 = (n_cur_80c[k+1]) - (n_cur_80c[k])
    factor = 1
    if y0 > 0:
        factor = -1
    ax00.arrow(xt, y0, dx*factor, dy0*factor,  color = 'tab:green', width = 1e-6, shape ='full', head_width = 2e-4, head_length = 1)


ax11.plot(V, n_cur_70_n, color = 'teal')
ax11.plot(V, n_cur_80_n, color = 'hotpink')
ax11.set_ylim(-0.004, 0.002)
ax11.set_xlim(-80,0)
ax11.set_title('Flow on the line')
ax11.set_xlabel('Membrane Vm')
#  ax11.set_ylabel('I [A]')
#  ax[1,1].legend()
ax11.set_title('End of NMDA plateau')

for k in range(0, len(V)-1, 6):
    xt = V[k]
    y0 = n_cur_70_n[k]
    dx = V[k+1] - V[k]
    dy0 = (n_cur_70_n[k+1]) - (n_cur_70_n[k])
    factor = 1
    if y0 > 0:
        factor = -1
    ax11.arrow(xt, y0, dx*factor, dy0*factor,  color = 'teal', width = 1e-6, shape ='full', head_width = 2e-4, head_length = 1)

for k in range(0, len(V)-1, 6):
    xt = V[k]
    y0 = n_cur_80_n[k]
    dx = V[k+1] - V[k]
    dy0 = (n_cur_80_n[k+1]) - (n_cur_80_n[k])
    factor = 1
    if y0 > 0:
        factor = -1
    ax11.arrow(xt, y0, dx*factor, dy0*factor,  color = 'hotpink', width = 1e-6, shape ='full', head_width = 2e-4, head_length = 1)

ax11.scatter(V[30], 0, color = 'teal', marker = 'o', label = 'Stable fixpoints')
ax11.scatter(V[84], 0, color = 'teal', marker = 'o')
ax11.scatter(V[59], 0, color = 'teal', marker = 'o', label = 'Stable fixpoints')
ax11.scatter(V[59], 0, color = 'white', marker = '.', label = 'Stable fixpoints')

ax11.scatter(V[71], 0, color = 'hotpink', marker = 'o', label = 'Stable fixpoints')
ax11.scatter(V[71], 0, color = 'white', marker = '.', label = 'Stable fixpoints')
ax11.scatter(V[8], 0, color = 'hotpink', marker = 'o', label = 'Stable fixpoints')

ax10.scatter(-61, 0, color = 'teal', marker = 'o', label = 'Stable fixpoints')
ax10.scatter(-78, 0, color = 'hotpink', marker = 'o', label = 'Stable fixpoints')
ax10.scatter(-14.3, 0, color = 'hotpink', marker = 'o')
ax10.scatter(-8.7, 0, color = 'teal', marker = 'o')
ax10.scatter(-27, 0, color = 'hotpink', marker = 'o', label = 'Stable fixpoints')
ax10.scatter(-27, 0, color = 'white', marker = '.', label = 'Stable fixpoints')
ax10.scatter(-35, 0, color = 'teal', marker = 'o', label = 'Stable fixpoints')
ax10.scatter(-35, 0, color = 'white', marker = '.', label = 'Stable fixpoints')

img_70 = np.load('./img/70_shift.npy')
img_80 = np.load('./img/80_shift.npy')
#  i80 = []
#  i70 = []
#  for i in range(30):
#      i80.append(img_80[0,:])
#      i70.append(img_70[0,:])
#  i80 = np.vstack(i80)
#  i70 = np.vstack(i70)
#  print(i70.shape)
#  print(img_80.shape)
#  img_70 = np.concatenate((i70, img_70), axis = 0)
#  img_80 = np.concatenate((i80, img_80), axis = 0)


Time = np.load('./img/time.npy')
#  Time = np.linspace(0,70,530)
#  print(Time.shape)
V = np.load('./img/V.npy')

x, y = np.meshgrid(V, Time)
x2, y2 = np.meshgrid(V[43:69], Time[72:])
x3, y3 = np.meshgrid(V[44:69], Time[72:])

ax2.imshow(img_80, aspect = 'auto', vmin = -.001, vmax = .0015, cmap = 'RdYlGn', extent = [min(V), max(V), 100, 0], interpolation = 'gaussian')
ax2.contour(x, y, (img_80), colors = 'tab:blue', levels = [0], linestyles = 'solid')
ax2.hlines(61,V[0], V[-1], color = 'tab:purple', ls = 'dashed', lw = 4)
ax3.hlines(61,V[0], V[-1], color = 'tab:purple', ls = 'dashed', lw = 4)
ax2.hlines(40,V[0], V[-1], color = '#E3735E', ls = 'dashed', lw = 4)
ax3.hlines(40,V[0], V[-1], color = '#E3735E', ls = 'dashed', lw = 4)
ax2.hlines(4,V[0], V[-1], color = 'tab:olive', ls = 'dashed', lw = 4)
ax3.hlines(4,V[0], V[-1], color = 'tab:olive', ls = 'dashed', lw = 4)
ax2.contour(x2, y2, (img_80[72:,43:69]), colors = '#FEEC9F', levels = [0], linestyles = 'dashed')
ax2.set_xlabel('V [mV]')
#  ax2.set_ylabel('I [mA]')
ax2.set_title('$\Delta E_K$ = 0mV')
ax2.legend(['$t_2$', '$t_1$', '$t_0$'], title = 'Landscape', loc = 4)
ax3.imshow(img_70, aspect = 'auto', vmin = -.001, vmax = .0015, cmap = 'RdYlGn', extent = [min(V), max(V), 100, 0], interpolation = 'gaussian')
#  ax3.imshow(img_70, aspect = 'auto', vmin = -.001, vmax = .003, cmap = 'binary_r')
print(x.shape)
print(img_70.shape)
ax3.contour(x, y, (img_70), colors = 'tab:blue', levels = [0], linestyles = 'solid')
print(x3.shape)
print(img_70[72:, 44:69].shape)
ax3.contour(x3, y3, (img_70[72:,44:69]), colors = '#FEEC9F', levels = [0], linestyles = 'dashed')
ax3.set_xlabel('V [mV]')
#  ax3.set_ylabel('I [mA]')
ax3.set_title('$\Delta E_K$ = 16mV')
ax2.set_yticks([])
ax3.set_yticks([])
tax3 = ax3.twinx()
tax3.set_ylim(100,0)
tax3.set_ylabel('Time [ms]')

import matplotlib as mpl

#  cbaxes = inset_axes(ax2, width="40%", height="5%", loc=3)
cb = plt.colorbar(plt.cm.ScalarMappable(norm = mpl.colors.Normalize(-.001, .0015), cmap = 'RdYlGn'), ax=ax3, ticks=[-.001,0,.0015], label = 'I [mA]')
cb.ax.axvline( 0, c = 'tab:blue')
ax00.text(-0.1, 1.05, 'A', fontweight = 'bold', transform = ax00.transAxes, fontsize = 20)
ax01.text(-0.1, 1.05, 'B', fontweight = 'bold', transform = ax01.transAxes, fontsize = 20)
ax10.text(-0.1, 1.05, 'C', fontweight = 'bold', transform = ax10.transAxes, fontsize = 20)
ax11.text(-0.1, 1.05, 'D', fontweight = 'bold', transform = ax11.transAxes, fontsize = 20)
ax2.text(-0.1, 1.05, 'E', fontweight = 'bold', transform = ax2.transAxes, fontsize = 20)
ax3.text(-0.1, 1.05, 'F', fontweight = 'bold', transform = ax3.transAxes, fontsize = 20)

ax2.spines[['right', 'top']].set_visible(True)
ax3.spines[['right', 'top']].set_visible(True)




#  fig.savefig('plot_something', dpi = 1000)

fig.savefig('FIG_3_with_colorbar.svg', dpi = 400)
fig.savefig('FIG_3_with_colorbar.pdf', dpi = 400)
plt.show()
