import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
from celluloid import Camera
from scipy.interpolate import CubicSpline
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


data_78 = np.load('IV_data_EK-70.0.npy')
data_80 = np.load('IV_data_EK-70.0.npy')
data_74 = np.load('IV_data_EK-80.0.npy')
data_64 = np.load('IV_data_EK-64.npy')
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

data = [data_80, data_78, data_74, data_64]

def kiv(data, idx):
    V = data[idx][:,0]
    inward = [2,3,6,8,10]
    inward  = np.sum(data[idx][:,inward], axis = 1 )*2*np.pi*1e-4
    cs = CubicSpline(V, inward)
    return cs

kd0 = kiv(data, 0)
kd3 = kiv(data, -1)


def IV(data, idx):
    V = data[idx][:,0]
    inward = [1, 4, 5]
    outward = [2,3,6,8,10]
    inward  = np.sum(data[idx][:,inward], axis = 1 )*2*np.pi*1e-4
    outward = np.sum(data[idx][:,outward], axis = 1)*2*np.pi*1e-4
    mg_block = 1/(1 + 2/3.57 *np.exp(-0.15*V))
    syn_act = 2.9e-3*mg_block*(V - 0)
    return V, outward + inward #, syn_act

V, ud0 = IV(data, 0)
V, ud3 = IV(data, -1)

def f_sat(v):
    return 1 / (1 + np.exp(-(v - 20)/2))

sdict = {"kind":1, "x2":0.01, "s2":0, 'A': 1.17, 'B':1.17}
def I_NMDA(sdict, V_ext, dt):
    tau1 = 4
    tau2 = 42
    g = 7e-3*1.2
    g = 1.5e-3*1
    Vnmda = 0
    mg = 2
    mg_block = 1.1/(1 + mg/3.57 *np.exp(-0.1*V))*(V - 0)

    dA = -sdict['A']/tau1
    dB = -sdict['B']/tau2
    sdict['A'] += dA*dt
    sdict['B'] += dB*dt
    syn_act = g*(sdict['B'] - sdict['A'])*mg_block*(V_ext - Vnmda)
    syn_act_2 = g*(sdict['B'] - sdict['A'])
    return syn_act, sdict, syn_act_2

sdict_a = {"kind":0 ,"s1":.01, "delay":0, 'A':2.6, 'B':2.6, 'weight':1}
def I_AMPA(sdict, V_ext, dt):
    tau1 = .5
    tau2 = 1.5
    Vnmda = 0
    g = -1.5e-3 #friday
    dA = -sdict['A']/tau1
    dB = -sdict['B']/tau2
    sdict['A'] += dA*dt
    sdict['B'] += dB*dt
    syn_act = sdict['weight']*g*(sdict['B'] - sdict['A'])
    return syn_act, sdict

mg_block = 1/(1 + 2/3.57 *np.exp(-0.1*V))*(V - 0)

Time = []
time = 0
dt = 0.1
img0 = np.zeros((len(V),1000))
img3 = np.zeros((len(V),1000))
for i in range(1000):
    if i > 50:
        _, sdict, I_nmda = I_NMDA(sdict, 0, dt)
        I_ampa, sdict_a = I_AMPA(sdict_a, 0, dt)
    else:
        I_nmda = 0
        I_ampa = 0
    time += dt
    Time.append(time)
    img0[:,i] = ud0 + I_nmda*mg_block*1.1 + I_ampa
    img3[:,i] = ud3 + I_nmda*mg_block + I_ampa

I_in = np.linspace(0,0.00001, 200)
fig, ax = plt.subplots(1,1, figsize = (6,5))
O0_bin = []
O3_bin = []
for I in I_in:
    mask0 = np.where(img0[:, 100] + I*V  > 0)
    mask3 = np.where(img3[:, 100] + I*V  > 0)
    O0_bin.append(V[mask0[0][0]])
    O3_bin.append(V[mask3[0][0]])

ax.errorbar(I_in, O0_bin, linestyle = 'dashdot', color = 'pink')
ax.errorbar(I_in, O3_bin, linestyle = 'dashdot', color = 'teal')
ax.legend([80, 74], title = 'EK')
ax.set_xlabel('AMPA strength input [mA]')
ax.set_ylabel('Resulting depolarization [mV]')
ax.set_title('Comparing non-linearity')
fig.savefig('Non-linear_comp')
plt.show()



sdict = {"kind":1, "x2":0.01, "s2":0, 'A': 1.17, 'B':1.17}
nmda = []
Time = []
time = 0
dt = 0.1
vm0 = []
vm3 = []
vm0line = []
vm3line = []
for i in range(1000):
    _, sdict, I_nmda = I_NMDA(sdict, 0, dt)
    time += dt
    Time.append(time)
    nmda.append(I_nmda)
    up = (ud0 + I_nmda*mg_block*1.15)
    dw = (ud3 + I_nmda*mg_block)
    cs_dw = CubicSpline(V, dw).roots()
    cs_up = CubicSpline(V, up).roots()
    cs_dw = np.delete(cs_dw, np.where(np.abs(cs_dw) > 100))
    cs_up = np.delete(cs_up, np.where(np.abs(cs_up) > 100))
    vm3.append(cs_dw)
    vm0.append(cs_up)
    cs_dw[cs_dw > 0] = -200
    cs_up[cs_up > 0] = -200
    vm3line.append(max(cs_dw))
    vm0line.append(max(cs_up))




x, y = np.meshgrid(V, Time)
#  print(len(V), len(Time))##.shape)
x22, y22 = np.meshgrid(V[39:69], Time[72:])
x32, y32 = np.meshgrid(V[44:69], Time[72:])
del(Time)
sdict = {"kind":1, "x2":0.01, "s2":0, 'A': 1.17, 'B':1.17}
nmda = []
Time = []
time = 0
dt = 0.1
fig = plt.figure(figsize = (15,10), layout = 'constrained')
#  fig = plt.figure(figsize = (15,9))
spec = fig.add_gridspec(2,9)
ax0 = fig.add_subplot(spec[0,0:4])
ax1 = fig.add_subplot(spec[1,0:4])
ax2 = fig.add_subplot(spec[:,4:6])
ax3 = fig.add_subplot(spec[:,6:8])
axcm = fig.add_subplot(spec[:,-1])
cbaxes = inset_axes(axcm, width="20%", height="95%", loc = 3)
cb = plt.colorbar(plt.cm.ScalarMappable(norm = mpl.colors.Normalize(-.5, .3), cmap = 'binary'), cax = cbaxes, ticks=[-.5,0,.3], orientation='vertical', label = 'Current (nA)')
ax2.spines[['right','left','bottom','top']].set_visible(True)
ax3.spines[['right','left','bottom','top']].set_visible(True)
axcm.spines[['right','left','bottom','top']].set_visible(False)
axcm.set_yticks([])
axcm.set_xticks([])

cbaxes = inset_axes(axcm, width="20%", height="95%", loc = 3)
camera = Camera(fig)
for i in range(1000):
    _, sdict, I_nmda = I_NMDA(sdict, 0, dt)
    time += dt
    Time.append(time)
    nmda.append(I_nmda)
    #  if i%5 == 0:
    if i%5 == 0:
        cb = plt.colorbar(plt.cm.ScalarMappable(norm = mpl.colors.Normalize(-.5, .3), cmap = 'binary'), cax = cbaxes, ticks=[-.5,0,.3], orientation='vertical', label = 'Current (nA)')

        #  fig = plt.figure(figsize = (15,10), layout = 'constrained')
        #  #  fig = plt.figure(figsize = (15,9))
        #  spec = fig.add_gridspec(2,9)
        #  ax0 = fig.add_subplot(spec[0,0:4])
        #  ax1 = fig.add_subplot(spec[1,0:4])
        #  ax2 = fig.add_subplot(spec[:,4:6])
        #  ax3 = fig.add_subplot(spec[:,6:8])
        #  axcm = fig.add_subplot(spec[:,-1])
        #  cbaxes = inset_axes(axcm, width="20%", height="95%", loc = 3)
        #  cb = plt.colorbar(plt.cm.ScalarMappable(norm = mpl.colors.Normalize(-.5, .3), cmap = 'binary'), cax = cbaxes, ticks=[-.5,0,.3], orientation='vertical', label = 'Current (nA)')
        #  ax2.spines[['right','left','bottom','top']].set_visible(True)
        #  ax3.spines[['right','left','bottom','top']].set_visible(True)
        #  axcm.spines[['right','left','bottom','top']].set_visible(False)
        #  axcm.set_yticks([])
        #  axcm.set_xticks([])

        ax0.plot(V, ud0 + I_nmda*mg_block*1.15, color = 'hotpink', zorder = 300)
        ax0.plot(V, ud3 + I_nmda*mg_block, color = 'teal', zorder = 300)
        ytick = np.linspace(-0.003, 0.005,9, endpoint = True)
        ax0.set_yticks(ytick, -ytick)
        for k in range(0, len(V)-1, 7):
            xt = V[k]
            y0 = ud0[k] + I_nmda*mg_block[k]*1.15
            y2 = ud3[k] + I_nmda*mg_block[k]
            dx = V[k+1] - V[k]
            dy0 = (ud0[k + 1] + I_nmda*mg_block[k+1]*1.15) - (ud0[k] + I_nmda*mg_block[k]*1.15)
            dy2 = (ud3[k + 1] + I_nmda*mg_block[k+1]) - (ud3[k] + I_nmda*mg_block[k])
            factor_2 = 1
            factor_0 = 1
            if y0 > 0:
                factor_0 = -1
            if y2 > 0:
                factor_2 = -1
            ax0.arrow(xt, y0, dx*factor_0, dy0*factor_0,  color = 'hotpink', width = 1e-6, shape ='full', head_width = 2e-4, head_length = .5)
            ax0.arrow(xt, y2, dx*factor_2, dy2*factor_2,  color = 'teal', width = 1e-6, shape ='full', head_width = 2e-4, head_length = .5)

        ax0.set_xlim(-70, 0)
        ax0.set_ylim(-.004,.003)
        ax0.set_yticks(np.arange(-0.004, 0.004, 0.001), np.round(np.arange(-.3, .5, .1),2)[::-1])
        ax0.hlines(0, -80, 0, linestyle = 'dashed', color = 'black', zorder = -100)
        ax0.set_xlabel('$V_m$ (mV)')
        ax0.set_ylabel('Current (nA)')
        ax0.set_title('I-V curve dynamics')
        ax0.legend(['0', '18'], title = r'$\Delta E_{K^+} \; (mV)$')
        if len(vm3[i]) == 3:
            ax0.scatter( vm3[i][0],[0], color = 'teal', marker = 'o', s = 100, zorder = 100)
            ax0.scatter( vm3[i][1],[0], color = 'teal', marker = 'o', s = 100, zorder = 100)
            ax0.scatter( vm3[i][1],[0], color = 'white', marker = 'o', s = 40, zorder = 200)
            ax0.scatter( vm3[i][2],[0], color = 'teal', marker = 'o', s = 100, zorder = 100)

        if len(vm0[i]) == 3:
            #  print(vm0[i][0])
            ax0.scatter( vm0[i][0],[0], color = 'hotpink', marker = 'o', s = 100, zorder = 100)
            ax0.scatter( vm0[i][1],[0], color = 'hotpink', marker = 'o', s = 100, zorder = 100)
            ax0.scatter( vm0[i][1],[0], color = 'white', marker = 'o', s = 40, zorder = 200)
            ax0.scatter( vm0[i][2],[0], color = 'hotpink', marker = 'o', s = 100, zorder = 100)

        if len(vm3[i]) == 1:
            ax0.scatter( vm3[i][0],[0], color = 'teal', marker = 'o', s = 100)

        if len(vm0[i]) == 1:
            ax0.scatter( vm0[i][0],[0], color = 'hotpink', marker = 'o', s = 100)

        #  ax0.scatter( vm3[i],[0], color = 'teal', marker = 'o', s = 100)
        #  ax0.scatter( vm0[i],[0], color = 'hotpink', marker = 'o', s = 100)

        if i < 100:
            ax1.plot(Time[:i], vm3line[:i], color = 'teal', lw = 2)
            ax1.plot(Time[:i], vm0line[:i], color = 'hotpink', lw = 2)
            ax1.scatter(Time[i], vm3line[i], color = 'teal', marker = 'o', s = 100)
            ax1.scatter(Time[i], vm0line[i], color = 'hotpink', marker = 'o', s = 100)
            #  ax1.set_xlim(0,dt*100)


        else:
            ax1.plot(Time[:i], vm3line[:i], color = 'teal', lw = 2)
            ax1.plot(Time[:i], vm0line[:i], color = 'hotpink', lw = 2)
            ax1.scatter(Time[i], vm3line[i], color = 'teal', marker = 'o', s = 100)
            ax1.scatter(Time[i], vm0line[i], color = 'hotpink', marker = 'o', s = 100)
            #  ax1.set_xlim(0,Time[i])

        ax1.set_xlim(0,100)

        ax1.set_ylim(-90, 0)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('$V_m$ (mV)')
        ax1.set_title('$V_m$ dynamics')

        ax2.imshow(img0.T, aspect = 'auto', vmin = -.001, vmax = .003, cmap = 'binary_r', extent = [min(V), max(V), 100, 0])
        ax2.contour(x, y, (img0.T), colors = 'white', levels = [0], linestyles = 'solid')
        ax2.contour(x22, y22, img0[39:69, 72:].T, colors = 'black', levels = [0], linestyles = 'dashed')
        ax2.set_xlabel('$V_m$ (mV)')
        #  ax2.set_ylabel('Time [ms]')
        ax2.set_title('$\Delta E_{K^+}$ = $0\;mV$')
        ax2.hlines(time,min(V), max(V), color = 'red')

        ax3.imshow(img3.T, aspect = 'auto', vmin = -.001, vmax = .003, cmap = 'binary_r', extent = [min(V), max(V), 100, 0])
        ax3.contour(x, y, (img3.T), colors = 'white', levels = [0], linestyles = 'solid')
        ax3.set_xlabel('$V_m$ (mV)')
        ax3.contour(x32, y32, img3[44:69,72:].T, colors = 'black', levels = [0], linestyles = 'dashed')
        #  ax3.set_ylabel('Time [mV]')
        ax3.set_title('$\Delta E_{K^+}$ = $18\;mV$')
        ax3.hlines(time,min(V), max(V), color = 'red')
        if i > 50:
            ax3.scatter( vm3line[i-50],time, color = 'teal', marker = 'o', s = 100, zorder = 100)
            ax2.scatter( vm0line[i-50],time, color = 'hotpink', marker = 'o', s = 100, zorder = 100)
            ax3.plot( vm3line[:i-50],np.array(Time[50:i]) + 0, color = 'teal', lw = 4)
            ax2.plot( vm0line[:i-50],np.array(Time[50:i]) + 0, color = 'hotpink', lw = 4)

        ax3.set_yticks([])
        #  ax2.set_yticks([])
        #  axy = ax3.twinx()
        #  axy.set_ylim(0,100)
        #  ax2.set_yticks(np.arange(0,120,20), np.arange(0,120,20)[::-1])
        ax2.set_ylabel('Time (ms)')
        #  cb.ax.axvline( 0, c = 'white')

        ax0.text(-0.1, 1.05, 'A', fontweight = 'bold', transform = ax0.transAxes, fontsize = 20)
        ax1.text(-0.1, 1.05, 'B', fontweight = 'bold', transform = ax1.transAxes, fontsize = 20)
        ax2.text(-0.1, 1.025, 'C', fontweight = 'bold', transform = ax2.transAxes, fontsize = 20)
        #  ax3.text(-0.1, 1.05, 'D', fontweight = 'bold', transform = ax3.transAxes, fontsize = 20)
        #  plt.show()
        #  exit()

        camera.snap()

animation = camera.animate()
animation.save('animation.mp4')


#
