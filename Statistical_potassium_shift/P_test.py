import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import tqdm
plt.style.use('K_PAPER')


def fnn(N):
    loc = np.random.randint(0,100,110)
    res = np.zeros(10)
    for i in range(110):
        res[loc[i]//10] += 1
    return res

res_sample = []
res_p = []
for i in range(1000):
    res_sample.append(fnn(1))
    res_p.append(np.random.poisson(11, 10))

res_sample = np.hstack(res_sample)
res_p = np.hstack(res_p)

#  arr = np.linspace(-np.pi, np.pi, 1000)
#  disp = np.radians(30)
#  plt.plot(arr, stats.norm.pdf(arr, 0, disp))
#  plt.plot(arr, stats.vonmises.pdf(arr, 15, loc =0))
#  plt.show()

def tuning_vm(arr, s = .3, shift = 0):
    shift = np.radians(shift)*2
    kappa = 1/np.power(np.deg2rad(s)*2, 2)
    arr_r = np.linspace(-np.pi,np.pi, 100)
    val = stats.vonmises.pdf(np.deg2rad(arr), kappa, loc = 0 + shift)
    val_r = stats.vonmises.pdf(arr_r, kappa, loc = 0 + shift)
    return val / np.max(val_r)

arr = np.linspace(-90, 90, 1000)
#  plt.plot(arr, tuning_vm(arr, 11, 0))
#  print(tuning_vm([2, 80], 11, 0))
#  rvs = stats.vonmises.rvs(kappa = 1/np.sqrt(np.deg2rad(13)/2), size = 10)
#  plt.scatter(rvs, stats.vonmises.pdf(rvs, kappa = 1/np.sqrt(np.deg2rad(13)/2)))

#  print(np.mean(tuning_vm(rvs, 13, 90)))
#  print(np.mean(tuning_vm(rvs, 13, 45)))
#  print(np.mean(tuning_vm(rvs, 13, 0)))

#  plt.show()
#  exit()
#

#  def tuning(arr, s = .3, shift = 0):
#      shift /= 90
#      arr = (arr + shift - 1e-7)%1
#      val = stats.norm.pdf(arr, 1, s)
#      return val / np.max(val)
#  arr = np.linspace(0,1,100)




def fn(N,s, shift):
    Ns = fnn(N)
    res = []
    #  synapses = np.random.uniform(0,1,1000)
    #  synapses = np.random.uniform(-np.pi,np.pi,1000)
    synapses = np.random.uniform(-90,90,1000)
    for N in Ns:
        if N < 2:
            continue
        pick = np.random.choice(synapses, int(N), replace = True)
        res.append(np.mean(tuning_vm(pick, s, shift)))
    return res



def fn_stacked(N, w, shift =0):
    ranbin = []
    for i in (range(10000)):
        ranbin.append(fn(N, w, shift))
    return np.hstack(ranbin)


def fn_normal(N,s = 11, shift = 0, S = 15):
    disp = 1/np.power(np.deg2rad(S)*2, 2)
    #  print(1/np.power(np.deg2rad(11)*2, 2))
    #  print(disp)
    N = np.random.poisson(N,1)[0]
    if N < 2:
        N = 2
    #  rvs = 1 - np.abs(np.random.normal(0, sigma, N))
    #  rvs = np.abs(np.random.normal(0, np.deg2rad(11)/2, N))
    rvs = stats.vonmises.rvs(kappa = disp, loc = 0, size = N)
    #  print(np.rad2deg(rvs)/2)
    #  return(np.mean(tuning_vm(rvs, s, shift)))
    return(np.mean(tuning_vm(np.rad2deg(rvs)/2, s, shift)))

def fn_normal_stacked(N, w, shift):
    ranbin = []
    for i in (range(10000)):
        ranbin.append(fn_normal(N, w, shift))
    return np.hstack(ranbin)

bins = np.linspace(0,1,100)
res1 = fn_normal_stacked(8, 11, 0)
kde = stats.gaussian_kde(res1)
res1_exp = np.sum(bins*kde(bins)*1/100)
#  res2 = fn_normal_stacked(8, 16.6, 0)
#  res3 = fn_normal_stacked(8, 30, 0)

res1_90 = fn_normal_stacked(8, 11, 90)
#  res2_90 = fn_normal_stacked(8, 16.6, 22)
#  res3_90 = fn_normal_stacked(8, 30, 22)

ran1 = fn_stacked(8, 11, 45)
kde = stats.gaussian_kde(ran1)
ran1_exp = np.sum(bins*kde(bins)*1/100)
#  ran2 = fn_stacked(8, 16, 0)
#  ran3 = fn_stacked(8, 30, 90)


fig, ax = plt.subplots(1,2, figsize = (10,6), sharey = True)
ax[0].hist(ran1, bins = bins, histtype = 'stepfilled', color = '#FF0000', alpha = 1, density = True, stacked = False, label = 'Mean tuning heterogeneous spines')
ax[0].text(-0.1, 1.05, 'E', fontweight = 'bold', transform = ax[0].transAxes, fontsize = 20)
#  ax[0].hist(ran2, bins = bins, histtype = 'step', density = True, stacked = False)
#  ax[0].hist(ran3, bins = bins, histtype = 'step', density = True, stacked = False)

ax[0].hist(res1, bins = bins, histtype = 'stepfilled', alpha = 1, density = True, stacked = False, color = '#27348B', label = 'Mean tuning clustered spines')
#  ax[0].hist(res2, bins = bins, histtype = 'stepfilled', alpha = .4, density = True, stacked = False, color = 'orange')
#  ax[0].hist(res3, bins = bins, histtype = 'stepfilled', alpha = .4, density = True, stacked = False, color = 'tab:green')
ax[0].set_ylabel('Probability')
ax[0].set_xlabel('Mean synaptic strength tuning')
ax[0].legend()
ax[0].set_title('Target prefered orientation')
ax[0].vlines(ran1_exp, 0,5, color = 'grey', linestyle = 'dashed')
ax[0].vlines(res1_exp, 0,5, color = 'grey', linestyle = 'dashed')

ax[1].hist(ran1, bins = bins, histtype = 'stepfilled', density = True, stacked = False, label = 'Mean tuning heterogeneous spines', color = '#FF0000', alpha = 1)
#  ax[1].hist(ran2, bins = bins, histtype = 'step', density = True, stacked = False)
#  ax[1].hist(ran3, bins = bins, histtype = 'step', density = True, stacked = False)

ax[1].hist(res1_90, bins = bins, histtype = 'stepfilled', alpha = 1, density = True, stacked = False, color = '#27348B', label = 'Mean tuning clustered spines')
#  ax[1].hist(res2_90, bins = bins, histtype = 'stepfilled', alpha = .4, density = True, stacked = False, color = 'orange')
#  ax[1].hist(res3_90, bins = bins, histtype = 'stepfilled', alpha = .4, density = True, stacked = False, color = 'tab:green')
ax[1].set_ylabel('Probability')
ax[1].set_xlabel('Mean synaptic strength tuning')
#  ax[1].legend([10, 15, 30], title = 'Tuning width')
ax[1].set_title('Target non-prefered orientation')
ax[0].set_ylim(0,5)
fig.suptitle(r'Mean tuning on 10 $\mu m$ dendrite section, clustered vs random')
#  fig.savefig('Ptest', dpi = 200)
fig.savefig('FIG_1E.svg', dpi = 400)
fig.savefig('FIG_1E.pdf', dpi = 400)
plt.show()
#  exit()





print((np.mean(res1) - np.mean(ran1))/np.mean(ran1))
print(res1_exp/ran1_exp)
#  exit()
#  print((np.mean(res2) - np.mean(ran2))/np.mean(ran2))
#  print((np.mean(res3) - np.mean(ran3))/np.mean(ran3))
#
#  print(np.mean(ran1))
#  print(np.mean(ran2))
#  print(np.mean(ran3))

bin_1  = []
bin_2 = []
bin_3  = []
shift = np.linspace(0, 90, 45)
#  shift = np.linspace(0, 90, 20)
bins = np.linspace(0,1,100)
for i in tqdm.tqdm(range(45)):
    #  res1 = fn_normal_stacked(8, 11, shift[i])
    #  kde = stats.gaussian_kde(res1)
    #  res1_exp = np.sum(bins*kde(bins)*1/100)
    print(shift[i])
    res1 = fn_normal_stacked(8, 11, shift[i])
    kde = stats.gaussian_kde(res1)
    res1_exp = np.sum(bins*kde(bins)*1/100)

    #  res2 = fn_normal_stacked(9, 20, shift[i])
    #  kde = stats.gaussian_kde(res2)
    #  res2_exp = np.sum(bins*kde(bins)*1/100)
    #  #
    #  res3 = fn_normal_stacked(9, 30, shift[i])
    #  kde = stats.gaussian_kde(res3)
    #  res3_exp = np.sum(bins*kde(bins)*1/100)
    #
    ran1 = fn_stacked(8, 11, shift[i])
    kde = stats.gaussian_kde(ran1)
    ran1_exp = np.sum(bins*kde(bins)*1/100)
    #
    #  ran2 = fn_stacked(9, 20, shift[i])
    #  kde = stats.gaussian_kde(ran2)
    #  ran2_exp = np.sum(bins*kde(bins)*1/100)
    #  #
    #  ran3 = fn_stacked(9, 30, shift[i])
    #  kde = stats.gaussian_kde(ran3)
    #  ran3_exp = np.sum(bins*kde(bins)*1/100)

    #  bin_1.append(np.mean(ran1)/(np.mean(res1)))
    #  bin_2.append(np.mean(ran2)/(np.mean(res2)))
    #  bin_3.append(np.mean(ran3)/(np.mean(res3)))
    bin_1.append(res1_exp/ran1_exp)
    print(bin_1[-1])
    #  bin_2.append(res2_exp/ran2_exp)
    #  bin_3.append(res3_exp/ran3_exp)

#  fig, ax =  plt.subplots(1,1, figsize = (10, 5))
#  ax.plot(shift, bin_1)
#  ax.plot(shift, bin_2)
#  ax.plot(shift, bin_3)
#  ax.legend([10, 15, 30], title = 'Tuning width')
#  ax.set_ylabel(r'$Factor = \frac{\mathbf{E}(C)}{\mathbf{E}(R)}$')
#  ax.set_xlabel(r'$\Delta^{\circ}$ from soma prefered orientation')
#  plt.show()
#
#
bin1 = np.array(bin_1)
np.save('P_bin_10', bin1)
#  bin2 = np.array(bin_2)
#  np.save('P_bin_20', bin2)
#  bin3 = np.array(bin_3)
#  np.save('P_bin_30', bin3)

#  exit()

