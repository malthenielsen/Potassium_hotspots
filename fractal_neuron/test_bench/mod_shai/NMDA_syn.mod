COMMENT
//****************************//
// Created by Alon Polsky 	//
//    apmega@yahoo.com		//
//		2007			//
//****************************//
ENDCOMMENT

TITLE NMDA synapse 

NEURON {
	POINT_PROCESS nmda
	USEION ca READ cai WRITE ica VALENCE 2
	NONSPECIFIC_CURRENT inmda 

	RANGE e ,gmax,inmda, tau1
	RANGE gnmda

	GLOBAL n, gama,tau2

}

UNITS {
	(nA) 	= (nanoamp)
	(mV)	= (millivolt)
	(umho) = (micromho)
	(mM)    = (milli/liter)
        F	= 96480 (coul)
        R       = 8.314 (volt-coul/degC)

}

PARAMETER {
	gmax=1	(umho)
	
    e= 0.0	(mV)
	tau1=42 (ms): Was 90	
	tau2=2	(ms)	
	gama=0.1 	(/mV) :Was 0.08
	dt (ms)
	v		(mV)
	n=0.56		(/mM) :was 0.25
}

ASSIGNED { 
	inmda		(nA)   
	gnmda		(umho)
	ica 		(nA)
	cai		(mM)	
}
STATE {
	A (umho)
	B (umho)
}

INITIAL {
      gnmda=0 
	A=0
	B=0
}

BREAKPOINT {
	
	SOLVE state METHOD cnexp
	
	gnmda=(A-B)/(1+n*exp(-gama*v))
	:gnmda=A-B
	inmda = gmax*gnmda * (v-e)
	ica = inmda/10
}

DERIVATIVE state {
	
	A'=-A/tau1
	B'=-B/tau2
	
}

NET_RECEIVE (weight) {
	gmax=weight
	state_discontinuity( A, A+ gmax)
	state_discontinuity( B, B+ gmax)
}

