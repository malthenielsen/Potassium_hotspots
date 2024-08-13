import numpy as np
from matplotlib import pyplot as plt
#  plt.style.use('science')
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#  rc('font',**{'family':['Times']})
#  rc('text', usetex=True)
import matplotlib
import glob
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 25})


left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

#  dek  = np.linspace(4,10, 100)
dek = np.arange(14,1,-1)
print(dek)
#  data_40 = np.load('./res_2024/res_0.4_8_2024.npy')
#  data_45 = np.load('./res_2024/res_0.45_8_2024.npy')
#  data_50 = np.load('./res_2024/res_0.5_8_2024.npy')
#  data_55 = np.load('./res_2024/res_0.55_8_2024.npy')
#  data_60 = np.load('./res_2024/res_0.6_8_2024.npy')
#  data_65 = np.load('./res_2024/res_0.65_8_2024.npy')
#  data_70 = np.load('./res_2024/res_0.7_8_2024.npy')
#  data_75 = np.load('./res_2024/res_0.75_8_2024.npy')
#  data_80 = np.load('./res/res_0.8_8_301.npy')
#  data_85 = np.load('./res/res_0.85_8_301.npy')
fnames = glob.glob('Nres/*')
img = []
val = []
for fname in fnames:
    val.append(float(fname.split('_')[1]))
    img.append(np.load(fname))

key = np.argsort(val)
img = np.vstack(img)[key,:].T

#  img = np.vstack([data_40, data_45, data_50, data_55, data_60, data_65, data_70, data_75, data_80, data_85])
#  print(img)
#  print(img.shape)
#  dek = np.linspace(0,18,100)
tp = []
for i in range(13):
    idx = np.argmin(np.abs(np.flipud(img[i,:]) + 30))
    tp.append(dek[idx])

fig, ax = plt.subplots(1,1, figsize = (10,8))
im = ax.imshow(np.flipud(img), cmap = 'plasma_r', extent = [2,14, 0, 1], aspect = 'auto', vmin = -40, vmax = -10)
for i in range(13):
    #  print(tp)
    ax.vlines(tp[i], i/13, (i+1)/13, color = 'floralwhite', lw = 2, ls = '--')

ax.set(xlabel = 'N synapses', ylabel = 'Mean synaptic activity (a.u.)')
ax.set_xticks([2,4,6,8,10,12,14], [2,4,6,8,10,12,14])
plt.colorbar(im, ax = ax, label = 'Peak voltage [mV]')
ax.set_ylim(.384, .84)
fig.savefig('S2_N_mean')
plt.show()
exit()


#  fig, ax = plt.subplots(figsize = (6,6))
fig = plt.figure(figsize = (8,6))
gs = fig.add_gridspec(1, 13)
ax1 = fig.add_subplot(gs[:,0])
ax2 = fig.add_subplot(gs[:,1:])
ax1.imshow(img[1:,:100//18], aspect = 'auto', extent = [0,2, 0, 1], cmap = 'plasma_r')
ax1.set_xticks([1], [0])
im = ax2.imshow(img[1:,6*(100//18):], aspect = 'auto', extent = [6,18, 0, 1], cmap = 'plasma_r')

ax1.set_yticks(np.arange(0.05,1,.2), np.round(np.arange(.4, .9, .1)[::-1], 2))
ax2.set_yticks([])
ax2.set_xlabel('$\Delta E_K$ [mV]')
ax1.set_ylabel('Mean Spine input [AU]')
#  ax2.text(-0.1, 1.05, 'F', fontweight = 'bold', transform = ax.transAxes, fontsize = 20)
for i in range(2,3):
    ax1.vlines(tp[-(i +1)], i/10,(i+1)/10, color = 'floralwhite', lw = 2, ls = '--')
for i in range(4,10):
    if i == 4:
        ax2.vlines(tp[-(i +1)] + .1, i/10,(i+1)/10, color = 'floralwhite', lw = 2, ls = '--')
    else:
        ax2.vlines(tp[-(i +1)], i/10,(i+1)/10, color = 'white', lw = 2, ls = '--')
ax2.legend(['Transition point'], loc = 1)
plt.colorbar(im, ax = ax2, label = 'Peak voltage [mV]')
plt.savefig('Transition_heat', dpi = 200)
plt.savefig('FIG_2E.svg', dpi = 400)
plt.savefig('FIG_2E.pdf', dpi = 400)

plt.show()

#  fig, ax = plt.subplots(1,1, figsize = (8,4))
#  ax.plot(dek, data_60 - data_60.min(), color = 'black')
#  ax.plot(dek, data_65 - data_65.min(), color = 'black', ls = '-.')
#  ax.plot(dek, data_55 - data_55.min(), color = 'black', ls = 'dashed')
#  ax.legend([.55, .6, .65], title = 'Avg. synaptic drive')
#  ax.set_xlabel('$\Delta E_K$ [mV]')
#  ax.set_ylabel('Peak height [mV]')
#  ax.fill_between(dek, 0, 7.5, color = '#FF5733', alpha = .2)
#  ax.fill_between(dek, 7.51, 18, color = '#0047AB', alpha = .2)
#  ax.text(0.5*(left+right), 0.5*(bottom+top), 'Non-Linear regime',
#          horizontalalignment='center',
#          verticalalignment='center',
#          fontsize=20, color='#FF5733',
#          transform=ax.transAxes)
#  ax.text(0.5*(left+right), 0.35*(bottom+top), 'Linear regime',
#          horizontalalignment='center',
#          verticalalignment='center',
#          fontsize=20, color='#0047AB',
#          transform=ax.transAxes)
#  fig.savefig('test.png')
#
#
#  plt.show()
#
