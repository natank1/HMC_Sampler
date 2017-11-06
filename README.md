# HMC_Sampler
Hybrid Monte Carlo using pytorch -python 2.7

This code is an implemntation of Hybrid Monte Carlo using pytorch. THere are three blocks:
1. Hamiltonian- The class that manages the Hamiltonian claculations.

2 default_potential- the potential that is calculated in case the user does not define it explicetly.

3  HMC_sampler - The main class that defines the sampler. It has a constructor in which one can set the initial position and velocity or only the reqruied dimension. The user needs to define the required sample_size and if he wishes to provide his own potential.

There are two working examples in hmc_tests  
