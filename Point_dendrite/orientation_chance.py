import numpy as np
from matplotlib import pyplot as plt
#plt.style.use('K_PAPER')
from scipy import stats
import tqdm
from scipy.interpolate import UnivariateSpline, CubicSpline
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl


def rm(x, N = 10):
    return np.convolve(x, np.ones(N)/N, mode='valid')

data = np.load('Spike_curves_2024.npy')
data[:, -6:] = 0
angles = np.linspace(0,90,200)
np.save('Spike_angles.npy', angles)
fig, ax = plt.subplots(1,2, figsize = (12,6))
#  dek = [0,4, 5, 6, 7, 8, 9, 10]
dek = [0, 6, 8, 10, 12, 14, 16, 18]

ax[0].set_ylabel('Chance for dendritic spike')
#  ax.set_ylabel('Entropy [bits]')
ax[0].set_xlabel('Angle from soma prefered')
max_height = []
half_width = []

fig2, AX = plt.subplots(1,1, figsize = (3.5,5))

def OSI(arr, ori):
    arr2 = arr[::-1]
    arr = np.append(arr2, arr[1:])
    ori2 = ori[::-1]*-1
    ori = np.append(ori2, ori[1:])
    #  plt.plot(ori, arr)
    #  plt.show()
    ori = np.deg2rad(ori)
    top = np.sum(arr * np.exp(2*1j*ori))
    F = np.abs(top)/np.sum(arr)
    #  return 1 - np.arctan2(np.imag(F), np.real(F))
    return F

for i in range(len(dek)):
    max_height.append(np.nanmax(data[i,:]))
    ax[0].plot(angles[:-4], rm(data[i,:],5), color = plt.cm.viridis_r((dek[i] + 2)/12))
    osi = OSI(data[i,:], angles)
    #  print(osi, print(dek[i]))
    half_width.append(osi)
    prob = rm(data[i,:],5)[0]
    #  print('dEK = ', dek[i], '      probability = ', prob)
    print('dEK = ', dek[i], '  OSI = ', osi)

    #  for j in range(data.shape[1]):
        #  if data[i,j] < max_height[-1]/2:

            #  half_width.append(j-1)
            #  break


    #  ax.plot(angles, entropy(data[i,:]))
    #  spl = UnivariateSpline(angles[:20], data[i,:20], k = 5)
    #  spl.set_smoothing_factor(.5)
    #  xs = np.linspace(0,40,1000)
    #  ax.plot(xs, spl(xs))
cbaxes = inset_axes(ax[0], width="40%", height="5%", loc = 5)
plt.colorbar(plt.cm.ScalarMappable(norm = mpl.colors.Normalize(vmin=-2, vmax=18), cmap = 'viridis_r'), cax=cbaxes, orientation='horizontal', label = '$\Delta E_K$')
#  ax[0].set_ylim(0,1)

#  ax[0].legend(dek, title = 'EK')
ax[0].set_title('Spike probability as function \n of stimulation angle')

#  dek = [0,4,5, 6, 7, 8, 9, 10]

#  fig2, ax2 = plt.subplots(1,1, figsize = (8,6))
AX2 = AX.twinx()
ax2 = ax[1].twinx()
ax[1].errorbar(dek, max_height, label = 'probability', color = 'red', marker = 'x', ls = ':'  , lw = 2, markersize = 10)
ax[1].set_ylim(.4, 1)
AX.errorbar(dek, max_height, label = 'probability', color = 'black', marker = 'o', ls = '-'  , lw = 2, markersize = 10)
AX.set_ylim(.0, 1)

ax2.errorbar(dek, half_width, label = 'FWHM', color = 'blue',marker = 'x', ls = ':', lw = 2, markersize = 10)
#  ax2.errorbar(dek, angles[half_width], label = 'FWHM', color = 'blue',marker = 'x', ls = ':', lw = 2, markersize = 10)
AX2.errorbar(dek, half_width, label = 'FWHM', color = 'grey',marker = 'o', ls = '-', lw = 2, markersize = 10)
ax[1].set_xlabel(r'$\Delta$$E_K$')
ax[1].set_ylabel('Probability at soma prefered', color = 'red')
AX.set_xlabel(r'$\Delta$$E_K$')
AX.set_ylabel('Peak spike probability', color = 'black')

ax2.set_ylabel('Orientation selectivity index', color = 'grey')
ax2.set_ylim(0, 1)
ax2.set_title('Probability distribution \n characteristics')
AX2.set_ylabel('Orientation selectivity index', color = 'grey')
AX2.set_ylim(0, 1)
AX2.set_title('Probability distribution \n characteristics')
#  fig2.savefig('p_dist_char')
ax[0].text(-0.1, 1.05, 'G', fontweight = 'bold', transform = ax[0].transAxes, fontsize = 20)
ax[1].text(-0.1, 1.05, 'H', fontweight = 'bold', transform = ax[1].transAxes, fontsize = 20)
AX.set_xticks([0, 6, 12, 18], [0, 6, 12, 18])
fig.savefig('Probability_4_spike', dpi = 200)
fig.savefig('FIG_2GH.svg', dpi = 400)
fig.savefig('FIG_2GH.pdf', dpi = 400)
plt.tight_layout()
fig2.savefig('FIG_2H.pdf', dpi = 200)
plt.show()
exit()


#  exit()
#  def tuning_vm(arr, s = .3, shift = 0):
#      shift = np.radians(shift)*2 #von mises is for the full circle we only look at the half
#      #  disp = 1/np.sqrt(np.deg2rad(s)/2)
#      #  disp = 1/np.power(np.deg2rad(15)*2, 2)
#      disp = 1/np.power(np.deg2rad(s)*2, 2) # relation from kappa to std is std**2 = 1/k
#      arr_r = np.linspace(-np.pi,np.pi, 1000)
#      val = stats.vonmises.pdf(arr, disp, loc = 0 + shift)
#      val_r = stats.vonmises.pdf(arr_r, disp, loc = 0 + shift)
#      #  plt.plot(arr_r, val)
#      #  plt.plot(arr_r, val_r/np.max(val_r))
#      #  plt.show()
#      return val / np.max(val_r)

def tuning_vm(arr, s = 11, shift = 0):
    shift = np.radians(shift)*2 #von mises is for the full circle we only look at the half
    kappa = 1/np.power(np.deg2rad(s)*2, 2) # relation from kappa to std is std**2 = 1/k
    #  print(kappa, 'kappa')
    arr_r = np.linspace(-np.pi,np.pi, 100)
    val = stats.vonmises.pdf(np.deg2rad(arr), kappa, loc = 0 + shift)
    val_r = stats.vonmises.pdf(arr_r, kappa, loc = 0 + shift)
    return val / np.max(val_r)


#  def create_weight_and_delay(regime, stim_alpha):
#      N = np.random.poisson(9,1)[0]
#      N_syn  = np.array([7,8,9,10,11,12,13])
#      N = int(np.random.choice(N_syn, 1))
#      weights = np.zeros(N)
#      for i in range(N):
#          if regime == 'clustered':
#              #  disp = 1/np.sqrt(np.radians(11/2))
#              #  rvs = stats.vonmises.rvs(kappa = disp, loc = 0, size = 1)[0]
#              #  weights[i] = tuning_vm(rvs, 11, stim_alpha)
#              disp = 1/np.power(np.deg2rad(15)*2, 2)
#              rvs = stats.vonmises.rvs(kappa = disp, loc = 0, size = 1)[0]
#              weights[i] = tuning_vm(rvs, 11, stim_alpha)
#          else:
#              #  rvs = np.random.uniform(-np.pi,np.pi)
#              #  weights[i] = tuning_vm(rvs, 11, stim_alpha)
#              rvs = np.random.uniform(-np.pi,np.pi)
#              rvs = np.random.uniform(-90,90)
#              weights[i] = tuning_vm(rvs, 11, stim_alpha)
#      return np.mean(weights), N

def create_weight_and_delay(regime, stim_alpha):
    N_syn  = np.array([7,8,9,10,11,12,13])
    N = int(np.random.choice(N_syn, 1))
    weights = np.zeros(N)
    #  weights2 = np.zeros(N)
    #  delays = np.random.poisson(30, N) + 200
    #  stim_alpha = np.deg2rad(stim_alpha)
    for i in range(N):
        if regime == 'clustered':
            disp = 1/np.power(np.deg2rad(15)*2, 2)
            rvs = stats.vonmises.rvs(kappa = disp, loc = 0, size = 1)[0]
            weights[i] = tuning_vm(np.rad2deg(rvs)/2, 11, stim_alpha)
        else:
            rvs = np.random.uniform(-np.pi,np.pi)
            #  rvs = np.random.uniform(-90,90)
            weights[i] = tuning_vm(rvs/2, 11, stim_alpha)
        #  rvs = np.random.uniform(-90,90)
        #      rvs = np.random.uniform(-180,180)
        #      weights[i] = tuning_vm(rvs, 11, stim_alpha)
        #  rvs = np.random.uniform(-180,180)
        #  weights2[i] = tuning_vm(rvs, 11, stim_alpha)
    #  print(np.round(weights,2))
    return  np.mean(weights), N





def fast_fit(w, N, intercept = 36.4, slope = -5.98):
    return intercept + slope*w*N

def fast_fit_old(w, N, intercept = 35, slope = -8.02):
    return intercept + slope*w*N

def fast_fit(w, N):
    #  return -47.8960*w -2.2518*N + 56.2755
    #  return -41.009*w -2.27*N + 55.41
    #  return -41.8960*w -2.2418*N + 57.2755 #w8
    return -38.4360*w -2.08818*N + 52.3255 #w85

#  print(fast_fit(.7, 10))
#  exit()
    

def sample_1k(stim_alpha):
    EK = np.zeros(1000)
    EK_old = np.zeros(1000)
    for i in range(1000):
        w, N = create_weight_and_delay('clustered', stim_alpha)
        EK[i] = fast_fit(w, N)
        #  EK_old[i] = fast_fit_old(w, N)
    return EK

def sample_all(resolution = 200):
    angles = np.linspace(0,90,resolution)
    Sav = np.zeros((resolution,10000))
    Sav_old = np.zeros((resolution,10000))
    for i, angle in tqdm.tqdm(enumerate(angles)):
        Sav[i,:] = sample_1k(angle)
        #  Sav[i,:], Sav_old[i,:] = sample_1k(angle)

    #  return Sav, angles
    angles = np.linspace(0,90,resolution)
    return Sav, Sav_old, angles


#  fig, ax = plt.subplots(1,1, figsize = (10, 7), sharey = True)
orientation = np.load('P_bin_10.npy')
orientation = np.nan_to_num(orientation, nan = 0)
cs = CubicSpline(np.linspace(0,90,45),orientation) 
orientation = cs(np.linspace(0,90,200))
print(orientation)
orientation /= np.nanmax(orientation)
print(orientation)
EK = [0,4, 6, 8, 10, 12, 14, 16, 18]
#  colors = ['tab:green', 'tab:blue', 'tab:red', 'orange']
Sav, Sav_old, angles = sample_all()
int_sav = []
for i, Ek in enumerate(EK):
    Sav_tmp = Sav.copy()
    #  Sav_old_tmp = Sav.copy()
    for j in range(200):
        Sav_slice = Sav[j,:].copy()
        Sav_slice[Sav_slice > Ek*orientation[j]] = 0
        Sav_slice[Sav_slice != 0] = 1
        Sav_tmp[j,:] = Sav_slice

        #  Sav_slice = Sav_old[j,:].copy()
        #  Sav_slice[Sav_slice > Ek*orientation[j]] = 0
        #  Sav_slice[Sav_slice != 0] = 1
        #  Sav_old_tmp[j,:] = Sav_slice

    Sav_tmp = np.sum(Sav_tmp, axis = 1)/10000
    #  Sav_old_tmp = np.sum(Sav_old_tmp, axis = 1)/1000
    int_sav.append(Sav_tmp)
    #  ax.plot(angles, Sav_tmp, label = Ek)
    #  ax.plot(angles, Sav_old_tmp, label = Ek, ls = 'dashed')

int_sav = np.vstack(int_sav)
np.save('Spike_curves_2024', int_sav)

#  ax.set_ylim(-.05,1)
#  Sav_tmp = Sav_r.copy()
#  Sav_tmp[Sav_r > 0] = 0
#  Sav_tmp[Sav_tmp != 0] = 1
#  Sav_tmp = np.sum(Sav_tmp, axis = 1)/1000
#  ax.plot(angles, Sav_tmp, label = 'Random')


#  ax.legend(title = 'Ek')
#  ax.set_ylabel('Chance for dendritic spike')
#  ax.set_xlabel('Angle from soma prefered')
#
#  plt.show()







