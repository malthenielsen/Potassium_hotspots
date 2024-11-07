import numpy as np
from matplotlib import pyplot as plt

def Na_V(Vr, ENa, m1, h1, dt):

    gbar = 0.12 #mS/cm2  # 	(pS/um2)	: 0.12 mho/cm2
    #  gbar = 600*1.1*0.459*1e-3
    #  gbar = 5
    #  gbar = 0
    gbar = 5.5
    #  gbar = 6.
    							
    tha  = -35 #:-30:-35	(mV)		: v 1/2 for act		(-42)
    qa   = 9.8 #:9	(mV)		: act slope		
    Ra   = 0.182#	(/ms)		: open (v)
    #  Ra   = ra#	(/ms)		: open (v)
    Rb   = 0.14#:0.124	(/ms)		: close (v)
    #  Rb   = rb#:0.124	(/ms)		: close (v)
    
    thi1  = -50	#(mV)		: v 1/2 for inact
    thi2  = -75#	(mV)		: v 1/2 for inact
    #  thi1  = -40	#(mV)		: v 1/2 for inact
    #  thi2  = -65#	(mV)		: v 1/2 for inact
    qi   = 5#	(mV)	        : inact tau slope
    thinf  = -65#	(mV)		: inact inf slope
    #  thinf  = -35#	(mV)		: inact inf slope
    qinf  = 6.2	#(mV)		: inact inf slope
    Rg   = 0.0091#	(/ms)		: inact (v)	
    Rd   =0.024	#(/ms)		: inact recov (v) 
    
    temp = 23#	(degC)		: original temp 
    q10  = 2.3	#		: temperature sensitivity

    Vr -= 25

    def trap0(Vr, th, a, q):
        if (np.abs((Vr - th)/q) > 1e-6):
            return a * (Vr - th) / (1 - np.exp(-(Vr - th)/q))
        else:
            return a*q
    a = trap0(Vr,tha,Ra,qa)
    b = trap0(-Vr,-tha,Rb,qa)
    
    tadj = q10**((38 - 23)/10)
    
    mtau = 1/tadj/(a+b)
    minf = a/(a+b)
    
    a = trap0(Vr,thi1,Rd,qi)
    b = trap0(-Vr,-thi2,Rg,qi)
    htau = 1/tadj/(a+b)
    hinf = 1/(1+np.exp((Vr-thinf)/qinf))
    
    dm =  (minf-m1)/mtau
    dh =  (hinf-h1)/htau
    m1 += dm*dt
    h1 += dh*dt
    Vr += 25
    syn = tadj*gbar*m1*m1*m1*h1*(Vr-ENa)
    cn = tadj*gbar*m1*m1*m1*h1
    return syn, m1, h1, cn


def K_V(Vr, EK, n1, dt):
    #  gbar = 0.12# mho/cm2
    #  gbar = 100*0.5*1e-3
    #  gbar = 2*.1*.5
    gbar = 2*.1
    #  gbar = .75
    							
    tha  = 25#	(mV)		: v 1/2 for inf
    qa   = 9#	(mV)		: inf slope		
    
    Ra   = 0.02	#(/ms)		: max act rate
    Rb   = 0.006#	(/ms)		: max deact rate	:0.002 before. Changed to 0.006 (Febrouary 2015) for smaller AHP.
    
    temp = 23	#(degC)		: original temp 	
    q10  = 2.3	#		: temperature sensitivity
    
    a = Ra * (Vr - tha) / (1 - np.exp(-(Vr - tha)/qa))
    b = -Rb * (Vr - tha) / (1 - np.exp((Vr - tha)/qa))

    tadj = q10**((38 - 23)/10)
    ntau = 1/tadj/(a+b)
    ninf = a/(a+b)
    dn =  (ninf-n1)/ntau
    n1 += dn*dt
    syn = tadj*gbar*n1*(Vr-EK)
    cn = tadj*gbar*n1
    return syn, n1, cn

def Cai(IHVA, ILVA, INMDA, Ca, dt):
    TCa = 121.4
    aCa = 20
    Ca += dt*( -aCa*(2*np.pi*1e-7*(IHVA + ILVA) - INMDA) - Ca/TCa) + np.random.normal(0, 3e-7, 1)[0];
    return Ca

def KCa_V(Vr, EK, n2, cai, dt):
    #  gbar = 0.03# mho/cm2
    gbar = 3*.1*7*1e-4
    gbar = 0.3*.1*2
    #  gbar = 0.1
    caix = 1	
                                    
    Ra   = 0.01#:0.01	(/ms)		: max act rate  
    Rb   = 0.02#:0.02	(/ms)		: max deact rate 
    temp = 23#	(degC)		: original temp 	
    q10  = 2.3#			: temperature sensitivity
    a = Ra * cai**caix
    b = Rb
    tadj = q10**((38 - temp)/10)
    ntau = 1/tadj/(a+b)
    ninf = a/(a+b)

    dn =  (ninf-n2)/ntau
    n2 += dn*dt
    syn = tadj*gbar*n2*(Vr-EK)
    cn = tadj*gbar*n2
    return syn, n2, cn

def Km_V(Vr, EK, n3, dt):
    #  gbar = 0.02
    #  gbar = 2.2*1.27*1e-4
    gbar = 0.1*.5*2
    #  gbar = 0.5
    #  gbar = 2
    tha  = -30#	(mV)		: v 1/2 for inf
    qa   = 9#	(mV)		: inf slope		
    
    Ra   = 0.001#	(/ms)		: max act rate  (slow)
    Rb   = 0.001#	(/ms)		: max deact rate  (slow)
    
    temp = 23#	(degC)		: original temp 	
    q10  = 2.3#			: temperature sensitivity
    
    a = Ra * (Vr - tha) / (1 - np.exp(-(Vr - tha)/qa))
    b = -Rb * (Vr - tha) / (1 - np.exp((Vr - tha)/qa))
    
    tadj = q10**((38 - temp)/10)
    ntau = 1/tadj/(a+b)
    ninf = a/(a+b)

    dn = (ninf-n3)/ntau
    n3 += dn*dt

    #  syn = gbar*tadj*n3*(Vr-EK)
    syn = gbar*n3*(Vr-EK)
    cn = gbar*n3
    return syn, n3, cn

#  n = .1
#  VR = np.linspace(-90, 2, 100)
#  I_bin = []
#  for v in VR:
#      for i in range(1000):
#          I, n, cn, = Km_V(v, -80, n, 0.01)
#      I_bin.append(I)
#
#  plt.plot(VR, I_bin)
#  plt.show()
#  exit()



#  def Ca_HVA(Vr, E_Ca, non, dt):
#      m6 = 1.0/( 1.0 + np.exp(-( Vr + 20.0)/9.0) )
#      #  return 6*0.0256867*(np.power(m6 , 2) )*(Vr - E_Ca), 0,6*0.0256867*(np.power(m6 , 2) );
#      return 1*0.0256867*(np.power(m6 , 2) )*(Vr - E_Ca), 0,1*0.0256867*(np.power(m6 , 2) );
#
#  def Ca_HVA(Vr, E_Ca, m0, h0, dt):
#      gbar = .12*.4
#      vshift = 20
#      temp = 23
#      q10 = 2.3
#      vmin = -120
#      vmax = 100
#
#      tadj = q10 **((38 - temp)/10)
#      #a = .055 * (-27 - Vr)/(np.exp((-27 - Vr)/3.8) - 1)
#      a = 0.5*(-27 - Vr)/(np.exp((-27-Vr)/3.8) - 1)
#      b = 0.1*np.exp((-75-Vr)/17)
#      mtau = 1/tadj/(a+b)
#      minf = a/(a+b)
#
#      a = 0.000457*np.exp((-13-Vr)/50)
#      b = 0.0065/(np.exp((-Vr-15)/28) + 1)
#
#      htau = 1/tadj/(a+b)
#      hinf = a/(a+b)
#      #  print(hinf, minf)
#
#      dm = (minf - m0)/mtau
#      dh = (hinf - h0)/htau
#      h0 += dh*dt
#      m0 += dm*dt
#      cn = gbar*tadj*m0*m0*h0
#      syn_act = cn *(Vr - E_Ca)
#      return syn_act, m0, h0, cn
#
def Ca_HVA(Vr, E_Ca, m0, h0, dt):
    #  gbar = .01*.4*5
    gbar = .01*0

    Vr += 20
    if Vr == -27:
        Vr = Vr+0.0001
    mAlpha =  (0.055*(-27-Vr))/(np.exp((-27-Vr)/3.8) - 1)
    mBeta  =  (0.94*np.exp((-75-Vr)/17))
    mInf = mAlpha/(mAlpha + mBeta)
    mTau = 1/(mAlpha + mBeta)
    hAlpha =  (0.000457*np.exp((-13-Vr)/50))
    hBeta  =  (0.0065/(np.exp((-Vr-15)/28)+1))
    hInf = hAlpha/(hAlpha + hBeta)
    hTau = 1/(hAlpha + hBeta)
    Vr -= 20

    dm = (mInf - m0)/mTau
    dh = (hInf - h0)/hTau
    h0 += dh*dt
    m0 += dm*dt
    cn = gbar*m0*m0*h0
    syn_act = cn *(Vr - E_Ca)
    return syn_act, m0, h0, cn
#
def Ca_LVA(Vr, E_Ca, m0, h0, dt):
    gbar = .012*10

    Vr += 0
    if Vr == -27:
        Vr = Vr+0.0001
    mAlpha =  (0.055*(-27-Vr))/(np.exp((-27-Vr)/3.8) - 1)
    mBeta  =  (0.94*np.exp((-75-Vr)/17))
    mInf = mAlpha/(mAlpha + mBeta)
    mTau = 1/(mAlpha + mBeta)
    hAlpha =  (0.000457*np.exp((-13-Vr)/50))
    hBeta  =  (0.0065/(np.exp((-Vr-15)/28)+1))
    hInf = hAlpha/(hAlpha + hBeta)
    hTau = 1/(hAlpha + hBeta)
    Vr -= 0

    dm = (mInf - m0)/mTau
    dh = (hInf - h0)/hTau
    h0 += dh*dt
    m0 += dm*dt
    cn = gbar*m0*m0*h0
    syn_act = cn *(Vr - E_Ca)
    return syn_act, m0, h0, cn

#  def Ca_LVA(Vr, E_Ca, m0, h0, dt):
#      gbar = .02*5
#
#      tadj = 2.3**((38-21)/10)
#
#      v = Vr
#      mInf = 1.0000/(1+ np.exp((v - -30.000)/-6))
#      mTau = (5.0000 + 20.0000/(1+np.exp((v - -25.000)/5)))/tadj
#      hInf = 1.0000/(1+ np.exp((v - -80.000)/6.4))
#      hTau = (20.0000 + 50.0000/(1+np.exp((v - -40.000)/7)))/tadj
#
#      dm = (mInf - m0)/mTau
#      dh = (hInf - h0)/hTau
#      h0 += dh*dt
#      m0 += dm*dt
#      cn = gbar*m0*m0*h0
#      syn_act = cn *(Vr - E_Ca)
#      return syn_act, m0, h0, cn



#  m0 = .3982061799898637
#  h0 = 0.5412170604331992
#  h00 = 0.5
#  m00 = 0.00024031171281892628
#  I_bin = []
#  IL_bin = []
#  VR = np.linspace(-90, 1, 100)
#  for V in VR:
#      for i in range(100):
#          sa, m0, h0, cn = Ca_HVA(V, 128, m0, h0, 0.01)
#          saL, m00, h00, cn = Ca_LVA(V, 128, m00, h00, 0.01)
#      I_bin.append(sa)
#      IL_bin.append(saL)
#
#  plt.plot(VR, I_bin)
#  plt.plot(VR, IL_bin)
#  plt.plot(VR, np.array(IL_bin) + np.array(I_bin))
#  plt.show()
#
#
#  exit()
    

#  def Ca_LVA(Vr, E_Ca, non, dt):
#      m6 = 1.0/( 1.0 + np.exp(-( Vr + 30.0)/9.0) )
#      #  return 6*0.0256867*(np.power(m6 , 2) )*(Vr - E_Ca), 0,6*0.0256867*(np.power(m6 , 2) );
#      return 1*0.0256867*(np.power(m6 , 2) )*(Vr - E_Ca), 0,1*0.0256867*(np.power(m6 , 2) );
#
def Im(Vr, EK, m2, dt):
    gImbar = 0.99 #(mS/cm2) 
    tadj = 2.3**((38-23)/10)

    mAlpha = 3.3e-3*np.exp(2.5*0.04*(Vr - -35))
    mBeta = 3.3e-3*np.exp(-2.5*0.04*(Vr - -35))
    mInf = mAlpha/(mAlpha + mBeta)
    mTau = (1/(mAlpha + mBeta))/tadj

    dm = (mInf - m2) / mTau
    m2 += dm*dt
    syn = gImbar*m2*(Vr - EK)
    cn = gImbar*m2
    return syn, m2, cn

def Ih(Vr, EK, m3, dt):
    gIhbar = 0.01 # ms/cm2
    if Vr == 154.9:
        Vr += 1e-6 
    mAlpha =  0.001*6.43*(Vr+154.9)/(np.exp((Vr+154.9)/11.9)-1)
    mBeta  =  0.001*193*np.exp(Vr/33.1)
    mInf = mAlpha/(mAlpha + mBeta)
    mTau = 1/(mAlpha + mBeta)
    dm = (mInf-m3)/mTau

    m3 += dm*dt
    syn = gIhbar*m3*(Vr - EK)
    cn = gIhbar*m3
    #  return 0*syn, 0*m3, 0*cn
    return syn, m3, cn

def K_Leak(Vr, E_k):
    return 0.05*.1*10*(Vr - E_k), 0.05*.1*10

def KA_V(Vr, EK, n4, l1, dt): 
    gbar = 0.8*.1
    gbar = 0.7
    gbar = .9 #monday
    #  gbar = 2
    q10 = 5
    nmin = 0.1
    lmin = 2
    qtl = 1

    zeta = -1.5+(-1)/(1+np.exp((Vr--40)/5))
    a = np.exp(1.e-3*zeta*(Vr-11)*9.648e4/(8.315*(273.16+38))) 
    betn = np.exp(1.e-3*zeta*0.55*(Vr-11)*9.648e4/(8.315*(273.16+38))) 

    qt=q10**((38-24)/10)
    ninf = 1/(1 + a)

    taun = betn/(qt*0.05*(1+a))
    if taun<nmin:
        taun=nmin

    a = np.exp(1.e-3*3*(Vr-11)*9.648e4/(8.315*(273.16+38)))
    linf = 1/(1+ a)
    taul = 0.26*(Vr+50)/qtl
    if taul<lmin/qtl:
        taul=lmin/qtl

    dn = (ninf - n4)/taun
    dl =  (linf - l1)/taul
    n4 += dn*dt
    l1 += dl*dt
    syn = gbar*n4*l1*(Vr - EK)
    cn = gbar*n4*l1
    return syn, n4, l1, cn

#  param = {'m1':0.219, 'h1':0.0817, 'n1':0.0008, 'n2':0.00049, 'n3':0.097, 'm2':0.0474, 'm3':0.0023}

def update_channels(param, Vr, E, dt):
    I_channel = {
            "Na_V": 0,
            "K_V" : 0,
            "KM_V" : 0,
            "Ca_HVA": 0,
            "Ca_LVA": 0,
            "K_Ca": 0,
            "Ih": 0,
            "K_Leak" : 0,
            "KA_V" :0,
            }

    gate_channel = {
            "Na_V": 0,
            "K_V" : 0,
            "KM_V" : 0,
            "Ca_HVA": 0,
            "Ca_LVA": 0,
            "K_Ca": 0,
            "Ih": 0,
            "K_Leak" : 0,
            "KA_V" :0,
            }

    I, m1, h1, cn = Na_V(Vr, E['Na'], param["m1"], param['h1'], dt)
    param["m1"] = m1
    param["h1"] = h1
    I_channel["Na_V"] = I
    gate_channel["Na_V"] = cn

    I, n1, cn = K_V(Vr, E['K'], param["n1"], dt)
    param["n1"] = n1
    I_channel["K_V"] = I
    gate_channel["K_V"] = cn

    I, n3, cn = Km_V(Vr, E['K'], param["n3"], dt)
    param["n3"] = n3
    I_channel["KM_V"] = I
    gate_channel["KM_V"] = cn

    #  I, _, cn = Ca_HVA(Vr, E['Ca'],0, dt)
    #  I_channel["Ca_LVA"] = I
    #  gate_channel["Ca_LVA"] = cn
    #
    #  I, _, cn = Ca_LVA(Vr, E['Ca'],0, dt)
    #  I_channel["Ca_HVA"] = I
    #  gate_channel["Ca_HVA"] = cn

    I, m0, h0, cn = Ca_HVA(Vr, E['Ca'],param['m0'], param['h0'], dt)
    param['m0'] = m0
    param['h0'] = h0
    I_channel["Ca_HVA"] = I
    gate_channel["Ca_HVA"] = cn

    I, m0, h0, cn = Ca_LVA(Vr, E['Ca'],param['m00'], param['h00'], dt)
    param['m00'] = m0
    param['h00'] = h0
    I_channel["Ca_LVA"] = I
    gate_channel["Ca_LVA"] = cn

    #  Ca = Cai(I_channel["Ca_LVA"], I_channel["Ca_HVA"], param["Inmda"], param["Ca"], dt)
    #  param["Ca"] = Ca
    #  I_channel["Ca"] = Ca
    #  gate_channel["Ca_LVA"] = cn
    #
    #  I, n2, cn = KCa_V(Vr, E['K'], param["n2"], param["Ca"], dt)
    #  param["n2"] = n2
    #  I_channel["K_Ca"] = I
    #  gate_channel["K_Ca"] = cn

    I, m3, cn = Ih(Vr, E['K'], param['m3'], dt)
    param['m3'] = m3
    I_channel["Ih"] = I*0
    gate_channel["Ih"] = cn*0

    I, cn = K_Leak(Vr, E['K'])
    I_channel["K_Leak"] = I
    gate_channel["K_Leak"] = cn

    I, n4, l1, cn = KA_V(Vr, E['K'], param["n4"], param['l1'], dt)
    param["n4"] = n4
    param["l1"] = l1
    I_channel["KA_V"] = I
    gate_channel["KA_V"] = cn

    I_tot = 0
    for key, value in I_channel.items():
        I_tot -= value

    return I_tot, param, I_channel, gate_channel



#  E = {'K': -60, 'Na': 56, 'Cl': -68, 'Ca':128};
def sim(Vr, tmax, E):
    param = {'m1':0.219, 'h1':0.0817, 'n1':0.0008, 'n2':0.00049, 'n3':0.097, 'm2':0.0474, 'm3':0.0023, "Ca":1e-6, "Inmda":0, 'n4':0.004427, 'l1': 0.9989138754237681, 'h0':0.3982, 'm0':0.5412, 'h00':0.3982, 'm00':0.5412}
    dt = 0.01
    ts = int(tmax//dt)
    I_channel = {
            "Na_V": 0,
            "K_V" : 0,
            "KM_V" : 0,
            "Ca_HVA": 0,
            "Ca_LVA": 0,
            "K_Ca": 0,
            "Ih": 0,
            "K_Leak" : 0,
            "Ca" :0,
            "KA_V" :0,

            }
    I_arr = np.zeros(ts)
    for i in range(1000):
        I, m1, h1, _ = Na_V(Vr, E['Na'], param["m1"], param['h1'], dt)
        param["m1"] = m1
        param["h1"] = h1
        I_channel["Na_V"] = I

        I, n1, _ = K_V(Vr, E['K'], param["n1"], dt)
        param["n1"] = n1
        I_channel["K_V"] = I

        I, n3, _ = Km_V(Vr, E['K'], param["n3"], dt)
        param["n3"] = n3
        I_channel["KM_V"] = I

        #  I, _, _ = Ca_HVA(Vr, E['Ca'],0, dt)
        #  I_channel["Ca_LVA"] = I

        #  I, _, _ = Ca_LVA(Vr, E['Ca'],0, dt)
        #  I_channel["Ca_HVA"] = I

        I, m0, h0, _ = Ca_HVA(Vr, E['Ca'],param['m0'], param['h0'], dt)
        param['m0'] = m0
        param['h0'] = h0
        I_channel["Ca_HVA"] = I

        I, m0, h0, _ = Ca_LVA(Vr, E['Ca'],param['m00'], param['h00'], dt)
        param['m00'] = m0
        param['h00'] = h0
        I_channel["Ca_LVA"] = I

        #  I, _, _ = Ca_LVA(Vr, E['Ca'],0, dt)
        #  I_channel["Ca_HVA"] = I

        Ca = Cai(I_channel["Ca_LVA"], I_channel["Ca_HVA"], param["Inmda"], param["Ca"], dt)
        param["Ca"] = Ca
        I_channel["Ca"] = Ca

        I, n2, _ = KCa_V(Vr, E['K'], param["n2"], param["Ca"], dt)
        param["n2"] = n2
        I_channel["K_Ca"] = I

        I, m3, _ = Ih(Vr, E['K'], param['m3'], dt)
        param['m3'] = m3
        I_channel["Ih"] = I

        I, _ = K_Leak(Vr, E['K'])
        I_channel["K_Leak"] = I

        I, n4, l1, _ = KA_V(Vr, E['K'], param["n4"], param['l1'], dt)
        param["n4"] = n4
        param["l1"] = l1
        I_channel["KA_V"] = I
        
    return I_channel #[inav, ikv, ika, ikis, icav, ica, inap, ikir]

#  param = {'m1':0.219, 'h1':0.0817, 'n1':0.0008, 'n2':0.00049, 'n3':0.097, 'm2':0.0474, 'm3':0.0023, "Ca":1e-6, "Inmda":0, 'n4':0.004427, 'l1': 0.9989138754237681}
#  VRS = np.arange(-90, 60)
#  I_bin = []
#  I_bin_70 = []
#  I_bin_80 = []
#  I_bin_90 = []
#  for Vr in VRS:
#      dt = 0.1
#      I, n4, l1 = KA_V(Vr, -60, param["n4"], param['l1'], dt)
#      param["n4"] = n4
#      param["l1"] = l1
#      I_bin.append(I)
#      I, n4, l1 = KA_V(Vr, -70, param["n4"], param['l1'], dt)
#      param["n4"] = n4
#      param["l1"] = l1
#      I_bin_70.append(I)
#      I, n4, l1 = KA_V(Vr, -80, param["n4"], param['l1'], dt)
#      param["n4"] = n4
#      param["l1"] = l1
#      I_bin_80.append(I)
#      I, n4, l1 = KA_V(Vr, -90, param["n4"], param['l1'], dt)
#      param["n4"] = n4
#      param["l1"] = l1
#      I_bin_90.append(I)
#
#  plt.plot(VRS,np.array(I_bin_90)*2*np.pi*1e-4)
#  plt.plot(VRS,np.array(I_bin_80)*2*np.pi*1e-4)
#  plt.plot(VRS,np.array(I_bin_70)*2*np.pi*1e-4)
#  plt.plot(VRS,np.array(I_bin   )*2*np.pi*1e-4)
#  plt.legend([-90, -80, -70, -60], title ='E_K')
#  plt.xlabel('mV')
#  plt.ylabel('pA')
#  plt.title('K A_type')
#  plt.savefig('IV_KAtype')
#
#  plt.show()
#
#
#
#
Vrs = np.linspace(-90, 5, 101)
def IV_plot(Vrs, Ek):
    E = {'K': -60, 'Na': 56, 'Cl': -68, 'Ca':128};
    E['K'] = Ek
    Sav = np.zeros((len(Vrs), 13))
    for i, Vr in enumerate(Vrs):
        I_dict = sim(Vr, 20, E)
        idx = 1
        for key, value in I_dict.items():
            Sav[i, idx:] = value
            idx += 1
            #  print(key, idx)
            #  if i == 1:
                #  print(key)
        #  print(i)
    Sav[:, 0] = Vrs

    #  np.save(f'../1D_flow/IV_data_EK{Ek}', Sav)

#  Vs = np.linspace(-80, -70, 20)
#  for i in Vs:
#      IV_plot(Vrs, i)


#  IV_plot(Vrs, -80)
#  IV_plot(Vrs, -78)
#  IV_plot(Vrs, -74)
#  IV_plot(Vrs, -64)
