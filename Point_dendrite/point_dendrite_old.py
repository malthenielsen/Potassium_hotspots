import numpy as np 
from matplotlib import pyplot as plt 
#  from channels import *
from Nassi_channels import *
import tqdm
from scipy import stats
#  plt.style.use('science')

'''I extrensic is taken from the formulation from the average Neuron model. Only change I've made is in the conductance of these receptors aswell as i've added a Mg Block to the NMDA channel'''
def I_ampa(sdict, V_ext, dt):
    tau1 = .5
    tau2 = 1.5
    Vnmda = 0
    #  g = -8e-7
    #  g = -3.5e-8
    #  g = -.7e-7
    g = -.5e-7 #friday
    #  g = 7e-6*1.2/4

    dA = -sdict['A']/tau1
    dB = -sdict['B']/tau2
    sdict['A'] += dA*dt
    sdict['B'] += dB*dt

    syn_act = sdict['weight']*g*(sdict['B'] - sdict['A'])*(V_ext - Vnmda)
    return syn_act, sdict

def I_nmda(sdict, V_ext, dt):
    tau1 = 4
    tau2 = 42
    #  g = 1e-6
    #  g = -7e-5
    #  g = -3.5e-7
    g = -2.9e-7 # friday

    #  g = -2e-7
    #  g = -4e-7
    #  g = -7e-7
    #  g = -3.e-7
    #  g = -2e-7

    Vnmda = 0
    mg = 2
    #  mg_block = -1.1/(1 + mg/3.57 *np.exp(-0.062*V_ext))
    mg_block = -1/(1 + 2/3.57 *np.exp(-0.1*V_ext))

    dA = -sdict['A']/tau1
    dB = -sdict['B']/tau2
    sdict['A'] += dA*dt
    sdict['B'] += dB*dt
    syn_act = -g*sdict['weight']*(sdict['B'] - sdict['A'])*mg_block*(V_ext - Vnmda)
    syn_act_2 = g*sdict['weight']*(sdict['B'] - sdict['A'])
    return syn_act, sdict, syn_act_2

def tuning_vm(arr, s = .3, shift = 0):
    shift = np.radians(shift)*2 #von mises is for the full circle we only look at the half
    disp = 1/np.sqrt(np.deg2rad(s)/2)
    arr_r = np.linspace(-np.pi,np.pi, 1000)
    val = stats.vonmises.pdf(arr, disp, loc = 0 + shift)
    val_r = stats.vonmises.pdf(arr_r, disp, loc = 0 + shift)
    return val / np.max(val_r)

def tuning_vm(arr, s = 11, shift = 0):
    shift = np.radians(shift)*2 #von mises is for the full circle we only look at the half
    kappa = 1/np.power(np.deg2rad(s)*2, 2) # relation from kappa to std is std**2 = 1/k
    #  print(kappa, 'kappa')
    arr_r = np.linspace(-np.pi,np.pi, 100)
    val = stats.vonmises.pdf(np.deg2rad(arr), kappa, loc = 0 + shift)
    val_r = stats.vonmises.pdf(arr_r, kappa, loc = 0 + shift)
    return val / np.max(val_r)

def create_nmda(delay, w):
    stim = {"kind":1, "x2":0.01, "s2":0, "delay":delay, 'A': 1.17, 'B':1.17, 'weight':w}
    return stim

def create_ampa(delay, w):
    stim = {"kind":0 ,"s1":.01, "delay":delay, 'A':2.6, 'B':2.6, 'weight':w}
    return stim

def create_stim_dict(delay, weight, N_burst = 1):
    ''' Each stimuli is first activated at a delay time, here we spawn stimuli from their earlier create functions'''
    all_stim = []
    for k in range(N_burst):
        for i in range(len(delay)):
            #  ran = np.random.randint(-5,5,1)[0]
            ran = 0
            all_stim.append(create_nmda(delay[i] + ran + 500*k, weight[i]))
            all_stim.append(create_ampa(delay[i] + ran + 500*k, weight[i]))
    return all_stim

def Isyn(stims, time, dt, V):
    ''' Function that monitors whitch synapses that are active at a given time, and add all active synapses contribution'''
    I_return = 0
    I_comp = np.zeros(2)
    I_cc = np.zeros(2)
    for stim in stims:
        delay = stim["delay"]
        if time > delay and time < delay + 200:
            if stim["kind"] == 1:
                syn_act, stim, syn_act_2 = I_nmda(stim, dt, V)
                I_comp[0] += syn_act
                I_cc[0] += syn_act_2
            else:
                syn_act, stim = I_ampa(stim, dt, V)
                I_comp[1] += syn_act
                I_cc[1] += syn_act/V
            I_return += syn_act
    return I_return, I_comp, I_cc

def delay_tracker(stims):
    del_bin = []
    for stim in stims:
        if stim["kind"] == 1:
            delay = stim["delay"]
            del_bin.append(delay)
    return del_bin


def dEk(dE, N, w, k_max):
    dt = 0.01
    gamma = .004
    #  gamma = .002*w
    dEkM = k_max
    return gamma*N*w - 1/100*dE/(dEkM - dE)

def simulation(param, E, delays, plot = True, change = False):
    dt = 0.01
    I_bin = []
    ch_bin = []
    V_bin = []
    time = 0
    V = -58
    A = 2*np.pi*(1e-4)*1e-3
    C = 2
    RT = []

    stims = create_stim_dict(delays, param['weights'], 3)
    N = 150000
    I_chan = np.zeros((11,N))
    I_g = np.zeros((11,N))
    I_syn_comp = np.zeros((2,N))
    I_syn_cc = np.zeros((2,N))
    I_exp = np.zeros((2,N))
    E_arr = np.zeros(N)
    labs = []
    dE = 0
    mean_w = np.mean(param['weights'])

    DEL = delay_tracker(stims)


    #  for i in tqdm.tqdm(range(N)):
    for i in (range(N)):

        I_SYN, I_comp, icc = (Isyn(stims, time, V, dt))# define the synaptic current
        param['Inmda'] = I_comp[0]
        I_CHANEL, param, I_dict, g_dict = update_channels(param, V, E, dt) # define the channel current
        gbar = 0
        Vinf = 0

        #  if time > np.min(delays) and change == True:
        #      E['K'] += dEk(dE, param['N_syn'], mean_w, param['k_max'])*dt
        #      dE     += dEk(dE, param['N_syn'], mean_w, param['k_max'])*dt
        #  if time == 500:
        if time > np.min(delays)+200 and change == True:
            E['K'] = E['K_c']

        E_arr[i] = E['K']
        idx = 0
        for key, value in g_dict.items():
            gbar += value*A
            if key[0] == 'N':
                Vinf += value * E['Na']*A
                I_g[idx, i] = value
            if key[0] == 'K':
                Vinf += value * E['K']*A
                I_g[idx, i] = value
            if key[0] == 'C':
                Vinf += value * E['Ca']*A
                I_g[idx, i] = value
            idx += 1
                #  print(value, key)

        Vinf += I_SYN
        Vinf += -1e-8*V
        Vinf = Vinf/gbar
        tau_m = C*A/gbar
        V = Vinf + (V - Vinf) * np.exp(-dt/tau_m) + np.sqrt(2*param['D']*dt)*np.random.normal(0,1)
        I_exp[0,i] = Vinf
        I_exp[1,i] = tau_m
        ''' The update is using Euler simulation where I add Langevin noise. The last part can be ignored for now maybe
        Basically I just solve c_m * dV/dt = I_Int*A + I_ext as dV = (I_int/c_m + I_syn/(A*C))*dt
        '''
        #  V += ((I_SYN) / (A*C) + I_CHANEL/C)*dt #+ np.sqrt(2*param['D']*dt)*np.random.normal(0,1)
        time += dt
        #Record some different tings
        V_bin.append(V)
        ch_bin.append(I_CHANEL)
        I_bin.append(I_SYN)
        RT.append(time)
        I_syn_comp[:,i] = I_comp
        I_syn_cc[:,i] = icc
        idx = 0
        for key, value in I_dict.items():
            I_chan[idx,i] = value
            idx += 1
            if i == 0:
                labs.append(key)

    RT = np.array(RT)
    ch_bin = np.array(ch_bin)
    V_bin = np.array(V_bin)
    weight = param['weights']

    return_dict = {'RT':RT, 'channels':ch_bin, 'V':V_bin, 'I_ext':I_syn_comp, 'indi_ch':I_chan, 'cc':I_syn_cc, 'EK':E_arr, 'delays':DEL, 'weights':weight, 'IG':I_g}



    #plotting
    if plot:

        fig, ax = plt.subplots(3, 1, figsize = (15,10), sharex = True)
        #  for i in range(I_g.shape[0] - 2):

        for i in range(3):
            ax[0].plot(RT/1000, I_g[i,:]*2*np.pi*1e-4, label = labs[i], linewidth = .5)
        #  ax[0].plot(RT/1000, I_exp[0,:], linewidth = .5)
        #  ax[0].plot(RT/1000, I_exp[1,:], linewidth = .5)
        #  ax[0].set_ylabel('pA')
        #  ax[0].set_xlabel('Time [s]')
        #  ax[0].set_ylim(-0.004, 0.004)
        #  ax[0].legend(loc = 1)
        ax[0].set_title('Channel Current by kind')

        ax[1].plot(RT/1000, I_syn_comp[0,:], label = 'NMDA', color = 'tab:red', linewidth = .5)
        ax[1].plot(RT/1000, I_syn_comp[1,:], label = 'AMPA', color = 'coral', linewidth = .5)

        ax[1].plot(RT/1000,ch_bin*A, label = 'Channels', color = 'tab:blue', linewidth = .5)
        ax[1].set_ylabel('pA')
        ax[1].set_ylim(-0.00001, 0.00001)
        ax[1].set_xlabel('Time [s]')
        ax[1].legend(loc = 1)
        ax[1].set_title('Current contribution by sources')


        ax[2].plot(RT/1000,V_bin, label = 'membrane potential', linewidth = .5)
        ax[2].scatter(delays/1000, np.ones_like(delays)*-55, marker = '|', color = 'black', s = 300, label = 'Stimulation')
        ax[2].set_ylabel('Vm [mV]')
        ax[2].set_xlabel('Time [s]')
        ax[2].legend(loc = 1)
        ax[2].set_title('Membrane dynamics')

        ax2 = ax[2].twinx()
        ax2.plot(RT/1000,E_arr, label = 'Ek', linewidth = .5)
        plt.show()

    return return_dict

#  def create_weight_and_delay(regime, stim_alpha):
#      #  N = np.random.poisson(9,1)[0]
#      #  N = np.random.poisson(9,1)[0]
#      N_syn  = np.array([7,8,9,10,11,12,13])
#      N = int(np.random.choice(N_syn, 1))
#      weights = np.zeros(N)
#      delays = np.random.poisson(80, N) + 200
#      for i in range(N):
#          if regime == 'clustered':
#              disp = 1/np.sqrt(np.radians(11/2))
#              rvs = stats.vonmises.rvs(kappa = disp, loc = 0, size = 1)[0]
#              weights[i] = tuning_vm(rvs, 11, stim_alpha)
#              #  print(':)')
#          else:
#              rvs = np.random.uniform(-np.pi,np.pi)
#              weights[i] = tuning_vm(rvs, 11, stim_alpha)
#      return delays, weights, N

def create_weight_and_delay(regime, stim_alpha, i):
    np.random.seed(i)
    N_syn  = np.array([7,8,9,10,11,12,13])
    N = int(np.random.choice(N_syn, 1))
    weights = np.zeros(N)
    weights2 = np.zeros(N)
    delays = np.random.poisson(80, N) + 200
    print(delays)
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
        rvs = np.random.uniform(-90,90)
        #      rvs = np.random.uniform(-180,180)
        #      weights[i] = tuning_vm(rvs, 11, stim_alpha)
        #  rvs = np.random.uniform(-180,180)
        weights2[i] = tuning_vm(rvs, 11, stim_alpha)
    #  print(np.mean(weights), stim_alpha)
    #  print(np.round(weights,2))
    return delays, weights*1, N



'''Param dict is all values the simulation needs to alter during its runtime, the dict is updated for each time step by the update_channels function'''


