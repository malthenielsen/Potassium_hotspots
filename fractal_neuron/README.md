The program is made up of three scripts:

- Create_segments_branch.py:
Functions that can create neuron segements: Soma, Trunk and Dendrites.
In here the relevant ionic channels is inserted and their currents set according
to the Shai model L5PCbiopysics5.

- Frac_prep.py:
Creates locations for synapses when running the script, aswell as assigning it a
orientation and computes the weight in regards to the orientaion we are
stimulating at. This stimulation orientation is given as a argument when calling
the script. 

- Branching_neuron.py
Main program, used to simulate the simple neuron. 
Run this to simulate the neuron with one cluster at dendrite 30, and multiple
clusters spread out
