import numpy as np
from matplotlib import pyplot as plt
#  plt.style.use('K_PAPER')
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
parser.add_argument('--initial', type=int, help='A string constant')

args = parser.parse_args()

decay = args.decay
alpha = args.alpha
center = args.center
Tstep = args.time
initial = args.initial

np.random.seed(1)

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

def tuning_vm(arr, s = .3, shift = 0):
    shift = np.radians(shift)*2 #von mises is for the full circle we only look at the half
    kappa = 1/np.power(np.deg2rad(s)*2, 2) # relation from kappa to std is std**2 = 1/k
    #  print(kappa, 'kappa')
    arr_r = np.linspace(-np.pi,np.pi, 100)
    val = stats.vonmises.pdf(np.deg2rad(arr), kappa, loc = 0 + shift)
    val_r = stats.vonmises.pdf(arr_r, kappa, loc = 0 + shift)
    #  plt.plot(arr_r, val_r)
    #  plt.scatter(np.deg2rad(arr), val)
    #  plt.show()
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
    rvs = stats.vonmises.rvs(kappa = disp, loc = 0, size = N)
    
    #  print(np.rad2deg(rvs)/2)
    #  return(np.mean(tuning_vm(rvs, s, shift)))
    return(tuning_vm(np.rad2deg(rvs)/2, s, shift))

def fn_random(N, s, shift):
    rvs = np.random.uniform(-90,90,N)
    #  return np.random.uniform(0,1,N)
    return(tuning_vm(rvs, s, shift))
    





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
    N = 10
    xsyn.append(np.random.randint(l*10*i, l*10*(i+1), N))
    ysyn.append(np.random.randint(0, int(circ*10),N))
    if not initial:
        xsyn.append(xsyn[-1]-1)
        ysyn.append(ysyn[-1]-1)
        xsyn.append(xsyn[-1]-1)
        ysyn.append(ysyn[-1]-1)
    if i == center:
        activity.append(fn_normal(N, 11, alpha))
        if not initial:
            activity.append(activity[-1])
            activity.append(activity[-1])
    else:
        activity.append(fn_random(N, 11, alpha))
        if not initial:
            activity.append(activity[-1])
            activity.append(activity[-1])

    t0 = np.random.randint(0, 52,1)
    fire.append(np.random.poisson(80,N)+t0)
    
    if not initial:
        t0 = np.random.randint(0, 50,1) + Tstep
        fire.append(np.random.poisson(80,N)+t0)

        t0 = np.random.randint(0, 50,1) + 2*Tstep
        fire.append(np.random.poisson(80,N)+t0)
    

xsyn = np.hstack(xsyn)
ysyn = np.hstack(ysyn)
activity = np.hstack(activity)
fire = np.hstack(fire)
print(sum(activity), 'sum activity')

if initial:
    np.save('fire_weight', fire)

x_local = np.random.randint(50*10,60*10,10)
y_local = np.random.randint(0,int(circ*10),10)
print(x_local, y_local)



N = 50000

def run(kind):
    grid = np.zeros((l*10*11, int(circ*10)))
    print(grid.shape)
    if initial:
        N  =1000000
    else:
        N = 500000

    measure = np.zeros((40,N))
    measure_local = np.zeros((10,N+2))

    measure_local[:,-2] = (x_local - 500)/100
    measure_local[:,-1] = y_local/(circ*10)
    sums = np.zeros(10)
    average = []
    dx = 1/10
    dy = 1/10
    dt = 0.002
    print(D*dt/(dx**2))
    print(fire)

    for i in (range(N)):
        #  for j in range(10):
            #  if kind == 1:
        grid[xsyn, ysyn] += 5000*activity*gauss(fire/dt, 65/dt, i)
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
        measure_local[:, i] = grid[x_local,y_local]
        #  grid[grid > 0] -= 0.001*dt
        #  grid[grid < 0] = 0
        #  grid -= 0.025*grid/(grid + 1.5)*dt
        grid -= decay/10*1e-8*grid*dt*5e4
        if i%50000 == 0:
            average.append(grid.mean(axis = 1))
    #  return measure
    time = N*dt 
    np.save('grid_local', measure_local)
    return grid.T, average, measure[:,::100]

measure_gauss, average, measure = run(1)
#  np.save('measure_NAK_new_29_short', measure)
if center == 5 and initial:
    np.save(f'measure_NAK_onoff_{int(decay)}_{Tstep}_{int(alpha)}_{initial}', measure)
if center == 5:
    np.save(f'measure_NAK_night_{int(decay)}_{Tstep}_{int(alpha)}_{initial}', measure)
    #  np.save(f'measure_NAK_inn_{int(decay)}_{Tstep}_{int(alpha)}', measure_inn)
else:
    np.save(f'measure_NAK_night_{int(decay)}_{Tstep}_none', measure)
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
