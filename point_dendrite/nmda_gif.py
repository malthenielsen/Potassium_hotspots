import numpy as np
from matplotlib import pyplot as plt
plt.style.use('science')
from celluloid import Camera

data_78 = np.load('IV_data_EK-78.npy')
data_80 = np.load('IV_data_EK-80.npy')
data_74 = np.load('IV_data_EK-74.npy')
data_70 = np.load('IV_data_EK-70.npy')
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

#  EK_vals = [90, 77, 70, 60]
#  EK_vals = [ 77, 60]
data = [data_80, data_78, data_74, data_70]

def IV(data, idx):
    V = data[idx][:,0]
    inward = [1, 4, 5]
    outward = [2,3,6,8,10]
    #  data[idx][:,1] *= 1
    #  data[idx][:,4] *= 1
    #  data[idx][:,5] *= 1
    #
    #  data[idx][:,3]  *= 2
    #  data[idx][:,6]  *= 2
    #  data[idx][:,8]  *= 10
    #  data[idx][:,10] *= 2
    #  data[idx][:,2]  *= .5
    #  data[idx][:,2] *= 1
    #  data[idx][:,outward] *= 1.5
    inward  = np.sum(data[idx][:,inward], axis = 1 )*2*np.pi*1e-4
    outward = np.sum(data[idx][:,outward], axis = 1)*2*np.pi*1e-4
    mg_block = 1/(1 + 2/3.57 *np.exp(-0.15*V))
    syn_act = 2.9e-3*mg_block*(V - 0)
    fig, ax = plt.subplots(1,1, figsize = (10,7))
    #  for n in outward:
    #      print(n)
    #      ax.plot(V, data[idx][:,n]*2*np.pi*1e-4, label = labels[n-1])
    #  ax.plot(V, np.sum(data[idx][:,outward], axis = 1)*2*np.pi*1e-4, label = 'Sum')
    #  ax.set_ylim(-.001,.01)
    #  ax.legend()
    #  plt.show()

    #  fig, ax = plt.subplots(1,1, figsize = (10,7))
    #  ax.plot(V,inward, label = 'Inward intrensic')
    #  ax.plot(V,outward, label = 'Outward intrensic')
    #  mask = np.where(np.round(syn_act,5) ==np.round(-outward+inward, 5))
    #  print(mask)
    #  ax.plot(V,syn_act, label = 'NMDA')
    ax.plot(V,outward+inward+syn_act + 1e-6*V, label = 'Sum of Intrensic and Extrensic')
    ax.hlines(0, min(V), max(V), color = 'black')
    #  ax.plot(V,(outward+inward) - 0e-4, label = 'Sum of Intrensic')
    #  ax.plot(V,(-inward) - 0e-4, label = 'sum of intrensic')
    #  ax.plot(V,(outward) - 0e-4, label = 'Sum of Intrensic')
    ax.set_xlim(-90, 0)
    ax.set_ylim(-.005,.005)
    ax.set_xlabel('Voltage [mV]')
    ax.set_ylabel('I [pA]')
    ax.legend()
    plt.show()
    #  return V, outward, inward, syn_act
    return V, outward + inward #, syn_act

#  V, out1,in1, sa1 = IV(data, 0)
#  V, out2,in2, sa2 = IV(data, 1)

#
#  fig, ax = plt.subplots(1,2, figsize = (10,7))
#  ax[0].plot(V, out1, color = 'tab:green', linestyle = 'dashed', label = 'Outward current')
#  ax[0].plot(V, out2, color = 'tab:green', linestyle = 'dashdot')
#  ax[0].plot(V, in1, color = 'tab:red', linestyle = 'dashed' , label = 'Inward current, Ek -70')
#  ax[0].plot(V, in2, color = 'tab:red', linestyle = 'dashdot', label = 'Inward current, Ek -78')
#  ax[0].plot(V, sa1, color = 'tab:blue', linestyle = 'dashed', label = 'Synaptic current')
#  ax[0].plot(V, sa2, color = 'tab:blue', linestyle = 'dashdot')
#  ax[1].plot(V, out1 + in1 + sa1, color = 'black', linestyle = 'dashed' , label = 'Res current, Ek -70')
#  ax[1].plot(V, out2 + in2 + sa2, color = 'black', linestyle = 'dashdot', label = 'Res current, Ek -78')
#  ax[1].plot(V, out1 + in1, color = 'teal', linestyle = 'dashed' , label = 'Res cur no NMDA, Ek -70')
#  ax[1].plot(V, out2 + in2, color = 'teal', linestyle = 'dashdot', label = 'Res cur no NMDA, Ek -78')
#  ax[0].legend()
#  ax[1].legend()
#  ax[0].set_xlim(-90, 0)
#  ax[1].set_xlim(-90, 0)
#  ax[0].set_ylim(-.01,.01)
#  ax[1].set_ylim(-.01,.01)
#  plt.show()

#  exit()
V, ud0 = IV(data, 1)
V, ud1 = IV(data, 1)
V, ud2 = IV(data, 2)
V, ud3 = IV(data, 2)

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
    #  mg_block = -1.1/(1 + mg/3.57 *np.exp(-0.062*V_ext))
    #  mg_block = -1.1/(1 + mg/3.57 *np.exp(-0.11*V_ext))
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

mg = 2
#  mg_block = 1.1/(1 + mg/3.57 *np.exp(-0.11*V))*(V - 0)
mg_block = 1.1/(1 + mg/3.57 *np.exp(-0.2*V))*(V - 0)
mg_block = 1/(1 + 2/3.57 *np.exp(-0.15*V))*(V - 0)
mg_block = 1/(1 + 2/3.57 *np.exp(-0.1*V))*(V - 0)

Time = []
time = 0
dt = 0.1
img0 = np.zeros((len(V),500))
img3 = np.zeros((len(V),500))
for i in range(500):
    #  sdict = {"kind":1, "x2":0.01, "s2":0, 'A': 1.17, 'B':1.17}
    if i > 50:
        _, sdict, I_nmda = I_NMDA(sdict, 0, dt)
        I_ampa, sdict_a = I_AMPA(sdict_a, 0, dt)
    else:
        I_nmda = 0
        I_ampa = 0
    time += dt
    Time.append(time)
    img0[:,i] = ud0 + I_nmda*mg_block + I_ampa
    img3[:,i] = ud2 + I_nmda*mg_block + I_ampa

I_in = np.linspace(0,0.00001, 200)
fig, ax = plt.subplots(1,1, figsize = (6,5))
O0_bin = []
O3_bin = []
for I in I_in:
    mask0 = np.where(img0[:, 100] + I*V  > 0)
    #  print(mask0[0][0])
    mask3 = np.where(img3[:, 100] + I*V  > 0)
    O0_bin.append(V[mask0[0][0]])
    O3_bin.append(V[mask3[0][0]])

#  ax.plot(img0[:, 100])
#  ax.plot(img3[:, 100])
#  print(O0_bin)
ax.errorbar(I_in, O0_bin, linestyle = 'dashdot', color = 'pink')
ax.errorbar(I_in, O3_bin, linestyle = 'dashdot', color = 'teal')
ax.legend([80, 74], title = 'EK')
ax.set_xlabel('AMPA strength input [mA]')
ax.set_ylabel('Resulting depolarization [mV]')
ax.set_title('Comparing non-linearity')
fig.savefig('Non-linear_comp')
plt.show()

#  exit()



x, y = np.meshgrid(V, Time)
del(Time)
sdict = {"kind":1, "x2":0.01, "s2":0, 'A': 1.17, 'B':1.17}
nmda = []
Time = []
time = 0
dt = 0.1
fig = plt.figure(figsize = (10,6), layout = 'constrained')
spec = fig.add_gridspec(2,4)
ax0 = fig.add_subplot(spec[0,0:2])  
ax1 = fig.add_subplot(spec[1,0:2])  
ax2 = fig.add_subplot(spec[:,2])  
ax3 = fig.add_subplot(spec[:,3])  

#  camera = Camera(fig)
for i in range(500):
    _, sdict, I_nmda = I_NMDA(sdict, 0, dt)
    time += dt
    Time.append(time)
    nmda.append(I_nmda)
#      if i%3 == 0:
#          ax0.plot(V, ud0 + I_nmda*mg_block, color = 'pink')
#          ax0.plot(V, ud2 + I_nmda*mg_block, color = 'teal')
#          for k in range(0, len(V)-1, 7):
#              xt = V[k]
#              y0 = ud0[k] + I_nmda*mg_block[k]
#              y2 = ud2[k] + I_nmda*mg_block[k]
#              dx = V[k+1] - V[k]
#              dy0 = (ud0[k + 1] + I_nmda*mg_block[k+1]) - (ud0[k] + I_nmda*mg_block[k])
#              dy2 = (ud2[k + 1] + I_nmda*mg_block[k+1]) - (ud2[k] + I_nmda*mg_block[k])
#              factor_2 = 1
#              factor_0 = 1
#              if y0 > 0:
#                  factor_0 = -1
#              if y2 > 0:
#                  factor_2 = -1
#              ax0.arrow(xt, y0, dx*factor_0, dy0*factor_0,  color = 'pink', width = 1e-6, shape ='full', head_width = 2e-4, head_length = .5)
#              ax0.arrow(xt, y2, dx*factor_2, dy2*factor_2,  color = 'teal', width = 1e-6, shape ='full', head_width = 2e-4, head_length = .5)
#
#
#
#          ax0.set_xlim(-60, 0)
#          ax0.set_ylim(-.003,.003)
#          ax0.hlines(0, -80, 0, linestyle = 'dashed', color = 'red')
#          ax0.set_xlabel('V [mV]')
#          ax0.set_ylabel('I [mA]')
#          ax0.set_title('IV curve Intrensic chanels and NMDA')
#          ax0.legend(['0', '10'], title = r'$\Delta$Ek')
#          ax1.plot(Time, nmda, color = 'black')
#          ax1.set_xlim(0,50)
#          ax1.set_ylim(0,.2e-2)
#          ax1.set_xlabel('Time [ms]')
#          ax1.set_ylabel('I [mA]')
#          ax1.set_title('NMDA dynamics')
#
#          ax2.imshow(img0.T, aspect = 'auto', vmin = -.001, vmax = .003, cmap = 'inferno', extent = [min(V), max(V), 50, 0])
#          ax2.contour(x, y, (img0.T), colors = 'white', levels = [0], linestyles = 'dashdot')
#          ax2.set_xlabel('V [mV]')
#          ax2.set_ylabel('I [mA]')
#          ax2.set_title('$\Delta$EK = 0mV')
#          ax2.hlines(time,min(V), max(V), color = 'red')
#
#          ax3.imshow(img3.T, aspect = 'auto', vmin = -.001, vmax = .003, cmap = 'inferno', extent = [min(V), max(V), 50, 0])
#          ax3.contour(x, y, (img3.T), colors = 'white', levels = [0], linestyles = 'dashdot')
#          ax3.set_xlabel('V [mV]')
#          ax3.set_ylabel('I [mA]')
#          ax3.set_title('$\Delta$EK = 10mV')
#          ax3.hlines(time,min(V), max(V), color = 'red')
#          camera.snap()

fig, ax = plt.subplots(1, 2, figsize = (5,6))
x, y = np.meshgrid(V, Time)
np.save('./fixpoints/img/time', Time)
np.save('./fixpoints/img/V', V)
x2, y2 = np.meshgrid(V[52:69], Time)
ax[0].imshow(img0.T, aspect = 'auto', vmin = -.001, vmax = .003, cmap = 'inferno', extent = [min(V), max(V), 50, 0])
ax[0].contour(x, y, (img0.T), colors = 'blue', levels = [0], linestyles = 'solid')
ax[0].contour(x2, y2, (img0[52:69, :].T), colors = 'green', levels = [0], linestyles = 'solid')
ax[0].set_xlabel('V [mV]')
ax[0].set_ylabel('I [mA]')
ax[0].set_title('EK = 80mV')
ax[0].legend('Nullcline', loc = 3)
ax[1].imshow(img3.T, aspect = 'auto', vmin = -.001, vmax = .003, cmap = 'inferno', extent = [min(V), max(V), 50, 0])
ax[1].contour(x, y, (img3.T), colors = 'white', levels = [0], linestyles = 'dashdot')
ax[1].set_xlabel('V [mV]')
ax[1].set_ylabel('I [mA]')
ax[1].set_title('EK = 70mV')
fig.savefig('IV_dynamical_heat')
plt.show()

np.save('./fixpoints/70_shift', img3.T)
np.save('./fixpoints/80_shift', img0.T)

#  np.save('rune_saturday/img0', img0)
#  np.save('rune_saturday/img3', img3)


#  animation = camera.animate()
#  animation.save('ani.wmf')


#
