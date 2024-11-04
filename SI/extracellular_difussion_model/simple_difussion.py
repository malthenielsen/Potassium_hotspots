import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
import time
from scipy import stats
import tqdm
from matplotlib.gridspec import GridSpec

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

#  fig, ax = plt.subplots(1,2,figsize = (10,6))
res_upper = []
res_lower = []
res_gauss = []
for t in time:
    res_upper.append(upper_tri(60, t, 20))
    res_lower.append(lower_tri(60, t, 20))
    res_gauss.append(gauss(50, 15, t))
#
#  ax[0].plot(time, np.cumsum(res_upper))
#  #  ax[0].plot(time, np.cumsum(res_lower))
#  #  ax[0].plot(time, np.cumsum(res_gauss))
#  ax[1].plot(time, (res_upper))
#  #  ax[1].plot(time, (res_lower))
#  #  ax[1].plot(time, (res_gauss))
#  plt.show()



l = 10 #um
r = 1.115 # um
circ = np.round(2*np.pi*r,1)

D = .76*10
dt = 0.01


#  grid[50, 35] = 10

xsyn = np.random.randint(l*10,l*10*2, 10)
ysyn = np.random.randint(0,70, 10)

fire = np.random.poisson(5000,10)


xmeasure = np.random.randint(l*10,l*10*2, 40)
ymeasure = np.random.randint(0,70,  40)

activity = np.random.uniform(.5,1,10)



N = 50000
def run(kind):
    grid = np.zeros((l*10*3, int(circ*10)))
    N = 50000
    measure = np.zeros((40,N))
    sums = np.zeros(10)

    for i in tqdm.tqdm(range(N)):
        #  if i in fire:
            #  xsyn = np.random.randint(l*10, l*10*2, 1)[0]
            #  ysyn = np.random.randint(0,70, 1)[0]
            #  print(xsyn, ysyn)
            #  grid[xsyn, ysyn] += 1
        for j in range(10):
            if kind == 1:
                grid[xsyn, ysyn] += activity[j]*gauss(fire[j], 100, i)
            if kind == 2:
                grid[xsyn, ysyn] += activity[j]*upper_tri(1000, i, fire[j])
            if kind == 3:
                grid[xsyn, ysyn] += activity[j]*lower_tri(1000, i, fire[j])
            #  sums[j] += upper_tri(1000, i, fire[j])
            #  sums[j] += lower_tri(1000, i, fire[j])
            #  sums[j] += gauss(fire[j], 300, i)


        Pgrid = np.pad(grid, 1, mode = 'wrap')
        Pgrid[0,:] = 0
        Pgrid[-1,:] = 0
        laplacian = np.zeros_like(Pgrid) 
        laplacian = (np.roll(Pgrid, 1, axis = 1) + np.roll(Pgrid, -1, axis = 1) + np.roll(Pgrid, 1, axis = 0) + np.roll(Pgrid, -1, axis = 0))
        laplacian -= 4*Pgrid
        grid += laplacian[1:-1, 1:-1]*D*dt
        grid += -grid*0.0001*dt
        for j in range(40):
            measure[j,i] = np.mean(grid[xmeasure[j-5:j+5], ymeasure[j-5:j+5]])
    return measure

measure_gauss= run(1)
measure_upper= run(2)
measure_lower= run(3)


fig = plt.figure(layout="constrained", figsize = (16,6))
gs = GridSpec(3, 8, figure=fig)
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[1, :2])
ax3 = fig.add_subplot(gs[2, :2])
ax = fig.add_subplot(gs[:, 2:])
ax1.plot(time, (res_upper), color = 'coral')
ax2.plot(time, (res_lower), color = 'dodgerblue')
ax3.plot(time, (res_gauss), color = 'limegreen')
ax1.set(yticks = [], xticks =[], title = 'Potassium input function')
ax2.set(yticks = [], xticks =[])
ax3.set(yticks = [], xticks =[])

colors = ['coral', 'dodgerblue', 'limegreen']

#  plt.show()
#  exit()


#  fig, ax = plt.subplots(1,1, figsize = (12,7))
for i in range(40):
    ax.plot(measure_upper[i] , alpha = .05, color = colors[0])
ax.plot(np.mean(measure_upper, axis = 0), color = colors[0], label = 'Upper Tri mean')
#  ax.fill_between(np.arange(0,N,1), np.mean(measure_upper, 0) - np.std(measure_upper, 0),  np.mean(measure_upper, 0) + np.std(measure_upper, 0), color = colors[0], alpha = .3)

for i in range(40):
    ax.plot(measure_lower[i],  alpha = .05, color = colors[1])
ax.plot(np.mean(measure_lower, axis = 0), color = colors[1], label = 'Lower Tri mean')
#  ax.fill_between(np.arange(0,N,1), np.mean(measure_lower, 0) - np.std(measure_lower, 0),  np.mean(measure_lower, 0) + np.std(measure_lower, 0), color = colors[1], alpha = .3)

for i in range(40):
    ax.plot(measure_gauss[i],  alpha = .05, color = colors[2])
ax.plot(np.mean(measure_gauss, axis = 0), color = colors[2], label = 'gauss Tri mean')
#  ax.fill_between(np.arange(0,N,1), np.mean(measure_gauss, 0) - np.std(measure_gauss, 0),  np.mean(measure_gauss, 0) + np.std(measure_gauss, 0), color = colors[2], alpha = .3)

ax.vlines(25000, 0, .02, color = 'grey', ls = '--')
ax.vlines(40000, 0, .02, color = 'grey', ls = '--', label = 'interval for next spike')
ax.set_xticks([0, 10000, 20000, 30000, 40000, 50000], [0, 100, 200, 300, 400, 500])
ax.legend()
ax.set_ylim(0,0.02)
ax.set_xlabel('Time')
ax.set_ylabel('Concentration [AU]')

fig.savefig('Well_mixed')
plt.show()




