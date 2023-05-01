
TITLE    AMPA synapse for nucleus accumbens model

: see comments below



NEURON {

	POINT_PROCESS AMPA

	RANGE gbar, tau_r, tau_d, scale, spkcnt, countflag, i, t1, ca_ratio, ical, itmp, qfact, ko_change, na_change, eErev

	NONSPECIFIC_CURRENT i

 	USEION cal WRITE ical VALENCE 2

    USEION k READ ko, ki WRITE ik VALENCE 1

    USEION na READ nao, nai WRITE ina VALENCE 1



}



UNITS {

	(nA) = (nanoamp)

	(mV) = (millivolt)

	(umho) = (micromho)

    FARADAY = 96489 (coul)
	R       = 8.314 (volt-coul/degC)

}



PARAMETER {

	gbar = 8.5e-4   (umho) 	: approx 0.5:1 NMDA:AMPA ratio (Myme 2003)

							:   with mg = 0, vh = -70, one pulse, NMDA = 300 pS

							:   here AMPA = 593 pS (NMDA set to Dalby 2003)

	tau_r = 2.2 	(ms)   	: Gotz 1997, Table 1 - rise tau

	tau_d = 11.5  	(ms)   	: Gotz 1997, Table 1 - decay tau

	

	Erev = 0    	(mV)   	: reversal potential, Jahn 1998

	saturate = 1.2 			: causes the conductance to saturate - matched to 

							:    Destexhe's reduced model in [1]

	qfact = 2				: convert 22 degC to 35 degC

	ca_ratio = 0.005			: ratio of calcium current to total current

							: Burnashev/Sakmann J Phys 1995 485:403-418

							: with Carter/Sabatini Neuron 2004 44:483-493

    ko_change = 1
    na_change = 1
  
}




ASSIGNED {

	g (umho)

	v (mV)   		: postsynaptic voltage

	itmp	(nA)	: temp value of current

	i (nA)   		: nonspecific current = g*(v - Erev)

	ical (nA)		: calcium current through AMPA synapse (Carter/Sabatini)

	ina (nA)		: sodium current through AMPA synapse 
	ik (nA)		    : potassium current through AMPA synapse 

	t1 (ms)

	

	y1_add (/ms)    : value added to y1 when a presynaptic spike is registered

	y1_loc (/ms)



	countflag		: start/stop counting spikes delivered

	spkcnt			: counts number of events delivered to synapse

	scale			: scale allows the current to be scaled by weight

					: so NetCon(...,2) gives 2*the current as NetCon(...,1)

    ki  (mM)
    ko  (mM)
    nai  (mM)
    nao  (mM)
    eErev (mV)
}




STATE { 

	y1 (/ms) 

	y2    			: sum of beta-functions, describing the total conductance

}



INITIAL {

  	y1_add = 0

	scale = 0

	spkcnt = 0

	countflag = 0

	t1 = 0

	y1_loc = 0

}



BREAKPOINT {

  	SOLVE betadyn METHOD cnexp

    eErev = -1000*R*(celsius+273.16) / (FARADAY) * log((ki + nai * na_change)/((ko*ko_change) + (nao))) 
	g = gbar * y2

  	itmp = scale * g * (v - eErev)

  	i = (1-ca_ratio) * itmp

  	ical = ca_ratio * itmp

    ina = 5 * itmp/10

    ik =  3 * itmp/5
  
    :printf("Ik %g, Ina %g, erev %g, v %g \n", ik, ina, eErev, v)
    :printf("ki %g, ko %g, nai %g, na0 %g \n", ki, ko, nai, nao)

}



DERIVATIVE betadyn {

	: dynamics of the beta-function, from [2]

	y1' = -y1 / (tau_d/qfact)

	y2' = y1 - y2 / (tau_r/qfact)

}



NET_RECEIVE( weight, y1_loc (/ms) ) {

	: updating the local y1 variable

	y1_loc = y1_loc*exp( -(t - t1) / (tau_d/qfact) )



	: y1_add is dependent on the present value of the local

	: y1 variable, y1_loc

	y1_add = (1 - y1_loc/saturate)



	: update the local y1 variable

	y1_loc = y1_loc + y1_add



	: presynaptic spike is finaly registered

	y1 = y1 + y1_add



	: store the spike time

	t1 = t



	spkcnt = spkcnt + 1



	scale = weight

}





