import numpy as np
from matplotlib import pyplot as plt
#  plt.style.use('K_PAPER')
from scipy import stats
import matplotlib as mpl


def tuning_vm(arr, s = .3, shift = 0):
    shift = np.radians(shift)*2 #von mises is for the full circle we only look at the half
    kappa = 1/np.power(np.deg2rad(s)*2, 2) # relation from kappa to std is std**2 = 1/k
    arr_r = np.linspace(-np.pi,np.pi, 100)
    #  val = stats.vonmises.pdf(np.deg2rad(arr), kappa, loc = 0 + shift)
    val_r = stats.vonmises.pdf(arr_r, kappa, loc = 0 + shift)
    return arr_r, val_r / np.max(val_r)

random = np.random.uniform(-90,90,15)
cluster = np.rad2deg(stats.vonmises.rvs(1/np.power(np.deg2rad(15)*2, 2), 0, size = 15))/2

fig, ax = plt.subplots(2,1, figsize = (8,6), sharex = True, sharey = True)
ax[0].text(-0.1, 1.15, 'D', fontweight = 'bold', transform = ax[0].transAxes, fontsize = 20)
clu = []
ran = []
for i in range(10):
    print(random[i])
    x, val = tuning_vm(0, 11, cluster[i])
    x = np.linspace(-90,90,100)
    if i == 0:
        clu.append(val)
        ax[0].plot(x, val, color = 'grey', label = 'Individual spine')
    else:
        clu.append(val)
        ax[0].plot(x, val, color = 'grey')

    x, val = tuning_vm(0, 11,random[i])
    x = np.linspace(-90,90,100)
    ax[1].plot(x, val, color = 'grey')
    ran.append(val)

clu = np.mean(np.vstack(clu),0)
ran = np.mean(np.vstack(ran),0)
ax[0].plot(x, clu, color = 'tab:blue', label = 'Mean spine response')
ax[0].set_title('Clustered')
ax[0].legend()
ax[1].plot(x, ran, color = 'tab:red')
#  x_val = np.linspace(-np.pi, np.pi, 11, endpoint = True)
#  xx_val = np.linspace(-90, 90, 11, endpoint = True)
#  ax[1].set_xticks(x_val, xx_val)
ax[1].set_title('Random')
ax[1].set_xlabel('$\Delta\Theta^\circ$ from target orientation')
fig.savefig('FIG_1D.svg', dpi = 400)
fig.savefig('FIG_1D.pdf', dpi = 400)
#
#
#
#
plt.show()
#
#  #  random90 = np.deg2rad(np.random.uniform(-90,90,10))
#  #  cluster90 = (stats.vonmises.rvs(1/np.sqrt(np.deg2rad(11)/2), np.pi/2, size = 10))
#  #  print(cluster)
#  #  print(cluster90)
#  #  exit()
#
## figure one
#  cs = plt.imread('wilson_bg.png')

fig, ax = plt.subplots(1,1, figsize = (5,6))
#  ax.imshow(cs, extent = [0,90, 0, 1], aspect = 'auto')
x = np.linspace(0,90,200)
ax.plot(x, x/90, color = 'red', label = 'Random')
ax.text(-0.1, 1.05, 'B', fontweight = 'bold', transform = ax.transAxes, fontsize = 20)
#  ax.plot(x, stats.gamma.cdf(x, 1, 0, scale = 10), color = 'black', label = 'correlated')
#  ax.plot(x, np.cumsum(stats.norm.pdf(x, 0, 22))/np.max(np.cumsum(stats.norm.pdf(x,0,22))), label = 'Correlated')
ax.plot(x, np.cumsum(stats.vonmises.pdf(np.deg2rad(x), 1/np.power(np.deg2rad(11)*2, 2)))/np.max(np.cumsum(stats.vonmises.pdf(np.deg2rad(x),(1/np.power(np.deg2rad(11)*2,2))))), label = 'Correlated')
ax.legend()
ax.set_xlabel(r'$\Delta^{\circ}$ from target orientation')
ax.set_ylabel('Fraction of spines')
fig.suptitle('Wilson figure 5F')
#  fig.savefig('Wilson', dpi = 200)
fig.savefig('FIG_1B.svg', dpi = 400)
fig.savefig('FIG_1B.pdf', dpi = 400)
plt.show()
#
#
## Pinwheel


fig, axs = plt.subplots(ncols=7, nrows=5, figsize = (8,12))
gs = axs[1, 2].get_gridspec()
cNorm = mpl.colors.Normalize(vmin=-1, vmax=1)

random = np.deg2rad(np.random.uniform(-90,90,10))
cluster = (stats.vonmises.rvs(1/np.power(np.deg2rad(15)*2, 2), 0, size = 10))
#  print(random/np.deg2rad(90))
#  print(cluster/np.deg2rad(90))
norm = np.deg2rad(90)

axbig0 = fig.add_subplot(gs[:, 0])
axbig0.vlines(1, 0,5, color ='black', linewidth = 2.5, zorder = 0)
axbig0.vlines(1, 0,5, color ='white', linewidth = 1.5, zorder = 0)
axbig0.scatter(np.ones(10), np.linspace(0,5,10), color = 'black', s = 110, zorder = 1)
axbig0.text(-0.1, 1.05, 'C', fontweight = 'bold', transform = axbig0.transAxes, fontsize = 20)
#  axbig0.scatter(np.ones(10), np.linspace(0,5,10), color = plt.cm.hsv((random)/np.deg2rad(90)), s = 100, zorder = 2, norm = cNorm)
axbig0.scatter(np.ones(10), np.linspace(0,5,10), cmap = 'hsv', c = random/np.deg2rad(90), s = 100, zorder = 2, norm = cNorm)

axbig1 = fig.add_subplot(gs[:, 3])
axbig1.vlines(1, 0,5, color ='black', linewidth = 2.5, zorder = 0)
axbig1.vlines(1, 0,5, color ='white', linewidth = 1.5, zorder = 0)
axbig1.scatter(np.ones(10), np.linspace(0,5,10), color = 'black', s = 110, zorder = 1)
#  axbig1.scatter(np.ones(10), np.linspace(0,5,10), color = plt.cm.hsv((cluster)/np.deg2rad(90)), s = 100, zorder = 2, norm = norm)
axbig1.scatter(np.ones(10), np.linspace(0,5,10), cmap = 'hsv', c = cluster/np.deg2rad(90), s = 100, zorder = 2, norm = cNorm)
#  axbig1.scatter(np.ones(10), np.linspace(0,5,10), color = plt.cm.hsv(np.rad2deg(cluster)), s = 100, zorder = 2)
print(np.rad2deg(cluster))
#  print(np.deg2rad(90))

x_line = np.linspace(-np.pi,np.pi,100)
for ax in fig.get_axes():
    ax.axis('off')
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

for i in range(5):
    pdf = stats.vonmises.pdf(x_line, 5, loc = random[i]*2)
    pdf_s = stats.vonmises.pdf(x_line, 5, loc = 0)
    axtmp_0 = fig.add_subplot(gs[i, 1:3])
    axtmp_0.plot(x_line, pdf,color = plt.cm.hsv((random[i] + np.pi/2)/(2*norm)))
    axtmp_0.plot(x_line, pdf_s,color = 'grey', linestyle = 'dashed')
    axtmp_0.fill_between(x_line[48:52], pdf[48:52], 0, color = 'grey', alpha = 0.9)
    pdf = stats.vonmises.pdf(x_line, 5, loc = cluster[i]*2)
    axtmp_1 = fig.add_subplot(gs[i, 4:6])
    print(cluster[i]/norm)
    axtmp_1.plot(x_line, pdf, color = plt.cm.hsv((cluster[i] + np.pi/2)/(2*norm)))
    axtmp_1.fill_between(x_line[48:52], pdf[48:52], 0, color = 'grey', alpha = 0.9)
    axtmp_1.plot(x_line, pdf_s,color = 'grey', linestyle = 'dashed')
    axtmp_0.text(-3, .7, f'{i + 1}')
    axtmp_1.text(-3, .7, f'{i + 1}')
    #  axtmp_1.axis('off')
    if i == 4:
        axtmp_1.spines.right.set_visible(False)
        axtmp_1.spines.top.set_visible(False)
        axtmp_1.yaxis.set_ticks_position('left')
        axtmp_1.xaxis.set_ticks_position('bottom')
        axtmp_1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[-90, -45, 0, 45, 90])

        axtmp_0.spines.right.set_visible(False)
        axtmp_0.spines.top.set_visible(False)
        axtmp_0.yaxis.set_ticks_position('left')
        axtmp_0.xaxis.set_ticks_position('bottom')
        axtmp_0.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[-90, -45, 0, 45, 90])

    else:
        axtmp_1.axis('off')
        axtmp_0.axis('off')

        if i == 0:
            axtmp_0.set_title('Random')
            axtmp_1.set_title('Correlated')

#  ax_cmap = fig.add_subplot(gs[:, -1:], projection = 'polar')
ax_cmap = fig.add_subplot(gs[:, -1:])
im = ax_cmap.imshow(np.linspace(-90,90,10).reshape(1,10), cmap = 'hsv')
plt.colorbar(im, ax = ax_cmap, shrink = 20, label =  'Synaptic orientation')
ax_cmap.set_visible(False)
fig.savefig('FIG_1C.svg', dpi = 400)
fig.savefig('FIG_1C.pdf', dpi = 400)


#  fig.savefig('soma', dpi = 200)

plt.show()
#  exit()

## Figure four

img = np.zeros((200,200))
img_r = np.zeros((200,200))
ki = 140
ko = 4

r2 = np.linspace(1.1,1.5,200)
dk = np.linspace(0, 1, 200)

for i in range(200):
    for j in range(200):
        img[i,j] = -26.7*np.log((ki - dk[i]) / (ko + dk[i]*1/(r2[j]**2 -1)))
        img_r[i,j] = -26.7*np.log((ki - dk[i]*5) / (ko + dk[i]*5*1/(r2[j]**2 -1)))

EK0 = -26.7*np.log(ki/ko)
fig, ax = plt.subplots(1,1, figsize = (6,5))
x, y = np.meshgrid(r2, dk)
CS = ax.contour(x, y, img - EK0, colors = ['tab:red', 'tab:red'], levels = [1.25, 5], linewidths = 3, linestyles = '--')
CS.collections[1].set_label('$\Delta E_K$ in random')
ax.clabel(CS, fontsize = 30, )
diff_img = - EK0 + img
mask = np.where((diff_img < 5) & (diff_img > 1.25),False, True)
diff_img_r = - EK0 + img_r
diff_img_r[mask] = np.nan
im = ax.imshow(diff_img_r, cmap = 'YlGn_r',extent = [r2[0], r2[-1], dk[-1], dk[0]], aspect = 'auto', vmin = 2, vmax = 18)
im_max = np.nanmax(diff_img_r)
im_min = np.nanmin(diff_img_r)
print(im_min, im_max)
ax.set_xlabel(r'Outer radii $r_2$ [$\mu m$]')
ax.set_ylabel(r'$\Delta k_i * Factor$ [mM]')
fig.suptitle('Change in extracellular K in clustered region')
ax.legend()
ax.text(-0.1, 1.05, 'I', fontweight = 'bold', transform = ax.transAxes, fontsize = 20)

plt.colorbar(im, ax = ax, label = r'$\Delta E_k$')
plt.tight_layout()
#  fig.savefig('Heatmap', dpi = 200)
fig.savefig('FIG_1I.svg', dpi = 400)
fig.savefig('FIG_1I.pdf', dpi = 400)

img = np.zeros((200,200))
img_r = np.zeros((200,200))
ki = 140
ko = 4

r2 = np.linspace(1.1,1.5,200)
dk = np.linspace(0, 1, 200)

for i in range(200):
    for j in range(200):
        img[i,j] = -26.7*np.log((ki - dk[i]) / (ko + dk[i]*1/(r2[j]**2 -1)))
        img_r[i,j] = -26.7*np.log((ki - dk[i]) / (ko + dk[i]*1/(r2[j]**2 -1)))

EK0 = -26.7*np.log(ki/ko)
fig, ax = plt.subplots(1,1, figsize = (6,5))
x, y = np.meshgrid(r2, dk)
CS = ax.contour(x, y, img - EK0, colors = ['tab:red', 'tab:red'], levels = [1.25, 5], linewidths = 3, linestyles = '--')
CS.collections[1].set_label('$\Delta E_K$ in random')
ax.clabel(CS, fontsize = 30, )
diff_img = - EK0 + img
mask = np.where((diff_img < 5) & (diff_img > 1.25),False, True)
diff_img_r = - EK0 + img_r
diff_img_r[mask] = np.nan
im = ax.imshow(diff_img_r, cmap = 'YlGn_r',extent = [r2[0], r2[-1], dk[-1], dk[0]], aspect = 'auto', vmin = 2, vmax = 18)
ax.set_xlabel(r'Outer radii $r_2$ [$\mu m$]')
ax.set_ylabel(r'$\Delta k_i * Factor$ [mM]')
fig.suptitle('Change in extracellular K in clustered region')
ax.legend()
ax.text(-0.1, 1.05, 'I', fontweight = 'bold', transform = ax.transAxes, fontsize = 20)

plt.colorbar(im, ax = ax, label = r'$\Delta E_k$')
plt.tight_layout()
#  fig.savefig('Heatmap', dpi = 200)
fig.savefig('FIG_1I2.svg', dpi = 400)
fig.savefig('FIG_1I2.pdf', dpi = 400)

plt.show()

