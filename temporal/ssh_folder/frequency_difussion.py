import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
import time
from scipy import stats
#  import tqdm
from matplotlib.gridspec import GridSpec

import argparse

parser = argparse.ArgumentParser(description="Set constants via command-line arguments.")
parser.add_argument('--decay', type=float, help='An integer constant')
parser.add_argument('--alpha', type=float, help='A string constant')
parser.add_argument('--center', type=int, help='A string constant')
parser.add_argument('--time', type=int, help='A string constant')

args = parser.parse_args()

decay = args.decay
alpha = args.alpha
center = args.center
Tstep = args.time

np.random.seed(101)

time = np.arange(0,100,.01)

def upper_tri(g,time,ts):
    h = 2/g
    slope = h/g
    if time - ts > 0:
        return max(h-slope*(time - ts), 0)
    else:
        return 0

def lower_tri(g,time,ts):
    h = 2/g
    slope = h/g
    if time - ts < 0:
        return 0
    else:
        if slope*(time - ts) > h:
            return 0
        else:
            return slope*(time - ts)

def gauss(mu, sigma, time):
    return stats.norm.pdf(time, mu, sigma)

def tuning_vm(arr, s = 11, shift = 0):
    shift = np.radians(shift)*2 #von mises is for the full circle we only look at the half
    kappa = 1/np.power(np.deg2rad(s)*2, 2) # relation from kappa to std is std**2 = 1/k
    #  print(kappa, 'kappa')
    arr_r = np.linspace(-np.pi,np.pi, 100)
    val = stats.vonmises.pdf(np.deg2rad(arr), kappa, loc = 0 + shift)
    val_r = stats.vonmises.pdf(arr_r, kappa, loc = 0 + shift)
    return val / np.max(val_r)

#  tuning_vm(np.random.uniform(0,180,10), 11, 0)
#  exit()

def fn_normal(N,s = 11, shift = 0, S = 15):
    disp = 1/np.power(np.deg2rad(S)*2, 2)
    #  print(1/np.power(np.deg2rad(11)*2, 2))
    #  print(disp)
    #  N = np.random.poisson(N,1)[0]
    if N < 2:
        N = 2
    #  rvs = 1 - np.abs(np.random.normal(0, sigma, N))
    #  rvs = np.abs(np.random.normal(0, np.deg2rad(11)/2, N))
    #  rvs = stats.vonmises.rvs(kappa = disp, loc = 0, size = N)
    
    #  print(np.rad2deg(rvs)/2)
    #  return(np.mean(tuning_vm(rvs, s, shift)))
    return(tuning_vm(np.deg2rad(11), s, shift))
    #  return rvs

def fn_random(N, s, shift):
    rvs = np.random.uniform(-180,180,1)
    #  return np.random.uniform(0,1,N)
    #  return rvs
    return(tuning_vm(np.deg2rad(11), s, rvs))

    



#  fig, ax = plt.subplots(1,2,figsize = (10,6))
res_upper = []
res_lower = []
res_gauss = []
for t in time:
    res_upper.append(upper_tri(60, t, 20))
    res_lower.append(lower_tri(60, t, 20))
    res_gauss.append(gauss(50, 15, t))


l = 10 #um
r = 1.115 # um
circ = np.round(2*np.pi*r,1)


D = 1.96
D = .76


#  grid[50, 35] = 10

#  xsyn = np.random.randint(l*10,l*10*10, 10)
#  ysyn = np.random.randint(0,70, 10)
#
#  fire = np.random.poisson(5000,10)
#
#
#  xmeasure = np.random.randint(l*10,l*10*2, 40)
#  ymeasure = np.random.randint(0,70,  40)
#
#  activity = np.random.uniform(.5,1,10)

#  res1 = fn_normal(800, 11, 5)
#  ran1 = fn_random(800)
xsyn = []
ysyn = []
activity = []
fire = [] 
#  alpha = 0
for i in range(1,11):
    #  N = np.random.randint(7,13,1)[0]
    Nsyn = 10
    alphas = [alpha]
    if i == center:
        disp = 1/np.power(np.deg2rad(15)*2, 2)
        rvs = stats.vonmises.rvs(kappa = disp, loc = 0, size = Nsyn)
    else:
        rvs = np.random.uniform(-180,180, Nsyn)

    for t0, stim_alpha in enumerate(alphas):
        weights = np.zeros(Nsyn)
        for idx in range(Nsyn):
            weights[idx] = tuning_vm(np.rad2deg(rvs[idx])/2, 11, stim_alpha)
                #  weights[idx] = tuning_vm(rvs, 11, stim_alpha)


        weight = np.mean(weights)
        delays = np.random.uniform(0,1,130)
        delays = np.where(delays < 1/1000*weight*80)[0]
        delays += 300*t0 + 50
        print(weight, i)
        weights = np.ones_like(delays)
        if len(delays) > 0:
            #  weights = np.ones(1)
            #  delays = np.ones(1)*(-50000)
            #  N = 1
        #  N = len(weights)
            N = len(delays)

            xsyn.append(np.ones(N)*np.random.randint(l*10*i, l*10*(i+1), N))
            ysyn.append(np.ones(N)*np.random.randint(0, int(circ*10),N))
            activity.append(np.ones(N))

            #  t0 = np.random.randint(0, 52,1)
            #  fire.append(np.random.poisson(30,N)+t0)
            fire.append(delays)
    

xsyn = np.hstack(xsyn).astype(int)
ysyn = np.hstack(ysyn).astype(int)
activity = np.hstack(activity)
fire = np.hstack(fire)
print(fire)
print(sum(activity), 'number of sites')


np.save(f'fire_{alpha}', fire)
#  exit()


N = 50000

def run(kind):
    grid = np.zeros((l*10*11, int(circ*10)))
    N = 200000
    measure = np.zeros((40,N))
    sums = np.zeros(10)
    average = []
    dx = 1/10
    dy = 1/10
    dt = 0.002
    print(D*dt/(dx**2))
    print(fire)

    for i in (range(N)):
        #  if N > 200000:
            #  dt = 0.005
        #  for j in range(10):
            #  if kind == 1:
        grid[xsyn, ysyn] += 7000*activity*gauss(fire/dt, 65/dt, i)
            #  if kind == 2:
            #      grid[xsyn, ysyn] += activity[j]*upper_tri(1000, i, fire[j])
            #  if kind == 3:
            #      grid[xsyn, ysyn] += activity[j]*lower_tri(1000, i, fire[j])

        Pgrid = np.pad(grid, 1, mode = 'wrap')
        Pgrid[0,:] = Pgrid[1,:]
        Pgrid[-1,:] = Pgrid[-2,:]
        #  laplacian = np.zeros_like(Pgrid)
        laplacian = (np.roll(Pgrid, 1, axis = 1) + np.roll(Pgrid, -1, axis = 1) + np.roll(Pgrid, 1, axis = 0) + np.roll(Pgrid, -1, axis = 0))
        laplacian -= 4*Pgrid
        grid += laplacian[1:-1, 1:-1]*D*dt*(1/(dx**2))
        for j in range(40):
            measure[j,i] = np.mean(grid[j*25, :])
        #  grid[grid > 0] -= 0.001*dt
        #  grid[grid < 0] = 0
        #  grid -= 0.025*grid/(grid + 1.5)*dt
        grid -= decay/10*1e-8*grid*dt*5e4
        if i%50000 == 0:
            average.append(grid.mean(axis = 1))
    #  return measure
    time = N*dt 
    return grid.T, average, measure[:,::100]

measure_gauss, average, measure = run(1)
#  np.save('measure_NAK_new_29_short', measure)
if center == 5:
    np.save(f'measure_NAK_onoff_{int(decay)}_{Tstep}_{int(alpha)}', measure)
    #  np.save(f'measure_NAK_inn_{int(decay)}_{Tstep}_{int(alpha)}', measure_inn)
else:
    np.save(f'measure_NAK_new_{int(decay)}_{Tstep}_none', measure)
    #  np.save(f'measure_NAK_inn_{int(decay)}_{Tstep}_none', measure_inn)
#  fig, ax = plt.subplots(3,1, figsize = (15,5), sharex = True)
#  ax[0].imshow(measure_gauss, aspect = 'auto')
#  ax[1].scatter(xsyn, activity)
#  for i in range(len(average)):
#      #  ax[2].plot(average[i])
#
#      ax[2].plot(average[i]/np.max(average[i]))
#  plt.show()
#  #  measure_upper= run(2)
#  #  measure_lower= run(3)
#  exit()
#
#
#
#
