import numpy as np
from matplotlib import pyplot as plt
#  plt.style.use('science')
import glob


def load(w,N):
    try:
        ret = np.load(f'./plane_plot_2/res_{w}_{N}_301.npy')
    except:
        ret = np.load(f'./plane_plot_2/res_{np.round(w,2)}_{N}_301.npy')
    return ret 

def load(w,N):
    try:
        ret = np.load(f'./res/res_{w}_{N}_301.npy')
    except:
        ret = np.load(f'./res/res_{np.round(w,2)}_{N}_301.npy')
    return ret 

def do_something(N):
    #  W = np.arange(0.1,1,0.1)
    #  w2 = np.linspace(.2, .7, 10)
    #  w3 = np.linspace(.2, .7, 10) + .1
    W = np.linspace(.2, .7, 10)
    w2 = np.linspace(.2, .7, 10) + .01
    w3 = np.linspace(.2, .7, 10) + .03
    W = np.hstack([W, w2, w3])
    #  W = np.linspace(.1, 1, 30)
    #  fnames = glob.glob('../plane_plot_2/*.npy')
    #  idx = []
    #  for fname in fnames:
    #      val = fname.split('_')[-3]
    #      idx.append(float(val))
    #
    #  W = np.sort(list(set(idx)))

    EK_jump = []

    for i, w in enumerate(W):
        data = load(w,N)
        EK = np.linspace(6, 18,len(data))
        EK_jump.append(EK[np.argmax(np.diff(data))])
    EK_jump = np.array(EK_jump)
    EK_jump[EK_jump == 6] = np.nan
    EK_jump[EK_jump == EK[-2]] = np.nan
    #  print(EK_jump)
    return W, np.array(EK_jump)

NV = np.array([7,8,9,10,11,12,13])
fig, ax = plt.subplots(figsize = (7,6))
A = []
B = []
for N in NV:
    W, EK = do_something(N)
    ax.scatter(W, EK)
    #  ax.plot(W, fit[2] + fit[1]*W)
    for i in range(len(W)):
        if not np.isnan(EK[i]):
            A.append([W[i], N, 1])
            B.append(EK[i])
ax.set(ylabel = '$\Delta E_K (mV)$', xlabel = 'Mean synaptic activity (a.u.)')
ax.legend(NV, title = 'Number of Synapses')
fig.savefig('Planar plot from the side')

A = np.matrix(A)
B = np.matrix(B).T

fit = (A.T * A).I * A.T * B


plt.figure(figsize = (8,8))
ax = plt.subplot(111, projection='3d')
ax.scatter(A[:,0], A[:,1], B, color='b')
#  print(A[:,0])
X, Y = np.meshgrid(np.linspace(0.15,.8, 10), np.linspace(6, 14, 10)) 
Z = np.zeros_like(X)
#  print(Z.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
ax.plot_wireframe(X,Y,Z, color='k')
ax.set(xlabel = 'Activity', ylabel = 'Number of spines', zlabel = r'$\Delta E_K$', zlim = (-10, 30))
plt.savefig('SI_FIG_1.svg', dpi = 400)
plt.savefig('SI_FIG_1.pdf', dpi = 400)

print(fit)

    

plt.show()



